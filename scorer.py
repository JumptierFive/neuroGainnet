"""Inference for the released gain networks.

The released weights are a single stacked tensor file (``gainnet_fp16.pt``)
holding the 2,700 networks (30 participants x 90 AAL regions) with prefix
``gain_net.`` stripped.

Architecture, applied to a 384-d sentence embedding:

    Linear(384, 100) -> ReLU -> Linear(100, 10) -> ReLU -> Linear(10, 1) -> Sigmoid

Example
-------
>>> from gainnet import GainScorer
>>> scorer = GainScorer("gainnet_fp16.pt")
>>> scorer.score(["A man is holding a bow and arrow."])   # (1, 90)
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ENCODER_NAME = "sentence-transformers/all-MiniLM-L12-v2"

#: AAL90 region order used throughout the release.
REGIONS: tuple[str, ...] = (
    "PreCG.L", "PreCG.R", "SFGdor.L", "SFGdor.R", "ORBsup.L", "ORBsup.R",
    "MFG.L", "MFG.R", "ORBmid.L", "ORBmid.R", "IFGoperc.L", "IFGoperc.R",
    "IFGtriang.L", "IFGtriang.R", "ORBinf.L", "ORBinf.R", "ROL.L", "ROL.R",
    "SMA.L", "SMA.R", "OLF.L", "OLF.R", "SFGmed.L", "SFGmed.R",
    "ORBsupmed.L", "ORBsupmed.R", "REC.L", "REC.R", "INS.L", "INS.R",
    "ACG.L", "ACG.R", "DCG.L", "DCG.R", "PCG.L", "PCG.R",
    "HIP.L", "HIP.R", "PHG.L", "PHG.R", "AMYG.L", "AMYG.R",
    "CAL.L", "CAL.R", "CUN.L", "CUN.R", "LING.L", "LING.R",
    "SOG.L", "SOG.R", "MOG.L", "MOG.R", "IOG.L", "IOG.R",
    "FFG.L", "FFG.R", "PoCG.L", "PoCG.R", "SPG.L", "SPG.R",
    "IPL.L", "IPL.R", "SMG.L", "SMG.R", "ANG.L", "ANG.R",
    "PCUN.L", "PCUN.R", "PCL.L", "PCL.R", "CAU.L", "CAU.R",
    "PUT.L", "PUT.R", "PAL.L", "PAL.R", "THA.L", "THA.R",
    "HES.L", "HES.R", "STG.L", "STG.R", "TPOsup.L", "TPOsup.R",
    "MTG.L", "MTG.R", "TPOmid.L", "TPOmid.R", "ITG.L", "ITG.R",
)


class GainScorer:
    """Score sentences with the released gain networks.

    Parameters
    ----------
    weights_path:
        Path to ``gainnet_fp16.pt`` (or ``gainnet_fp32.pt``).
    device:
        Torch device, ``"cpu"`` by default.
    """

    def __init__(self, weights_path: str = "gainnet_fp16.pt", device: str = "cpu"):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"{weights_path} not found. Download it from the Hugging Face repo "
                "and pass its path explicitly."
            )
        blob = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.W1 = blob["W1"].to(device).float()      # (2700, 100, 384)
        self.b1 = blob["b1"].to(device).float()      # (2700, 100)
        self.W2 = blob["W2"].to(device).float()      # (2700, 10, 100)
        self.b2 = blob["b2"].to(device).float()      # (2700, 10)
        self.W3 = blob["W3"].to(device).float()      # (2700, 10)
        self.b3 = blob["b3"].to(device).float()      # (2700,)
        self.device = device
        self.subject = blob.get("subject", None)
        self.region = blob.get("region", None)
        self._encoder = None

    # -- encoding ---------------------------------------------------------
    def _encode(self, sentences: Sequence[str], batch_size: int = 32) -> torch.Tensor:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(ENCODER_NAME, device=self.device)
        out = []
        for i in range(0, len(sentences), batch_size):
            chunk = list(sentences[i : i + batch_size])
            out.append(
                self._encoder.encode(
                    chunk,
                    convert_to_tensor=True,
                    normalize_embeddings=False,   # the networks were trained on unnormalised embeddings
                )
            )
        return torch.cat(out, 0).to(self.device).float()

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def score_embeddings(self, x: torch.Tensor) -> np.ndarray:
        """Map (n, 384) sentence embeddings to (n, 90) region gains."""
        x = x.to(self.device).float()
        h = torch.relu(torch.einsum("bd,sod->sbo", x, self.W1) + self.b1.unsqueeze(1))
        h = torch.relu(torch.einsum("sbo,sjo->sbj", h, self.W2) + self.b2.unsqueeze(1))
        out = torch.einsum("sbj,sj->sb", h, self.W3) + self.b3.unsqueeze(1)
        # weights are stacked subject-major: (30, 90) -> mean over participants
        n_sent = out.shape[1]
        per_region = torch.sigmoid(out).reshape(30, 90, n_sent).mean(0)   # (90, n)
        return per_region.transpose(0, 1).cpu().numpy()                   # (n, 90)

    def score(self, sentences: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Map sentences to (n, 90) region gains (mean over 30 participants)."""
        sentences = list(sentences)
        if not sentences:
            return np.zeros((0, len(REGIONS)), dtype=np.float32)
        return self.score_embeddings(self._encode(sentences, batch_size))

    def score_mean(self, sentences: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Map sentences to (n,) gains averaged over all 90 regions."""
        return self.score(sentences, batch_size).mean(1)


def weights_for(texts: Sequence[str], gains: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Turn per-sentence gains into softmax weights.

    Parameters
    ----------
    texts:
        Sentences of one paragraph (unused; kept for API symmetry).
    gains:
        (n,) or (n, 90) gains for those sentences.
    tau:
        Temperature in within-paragraph standard-deviation units. The value
        used in the paper (``tau~2`` after z-scoring within the paragraph)
        corresponds to roughly ``tau=1.0`` on raw gains for short paragraphs;
        scan the value for your data.
    """
    g = np.asarray(gains, dtype=np.float64)
    if g.ndim > 1:
        g = g.mean(1)
    z = (g - g.mean()) / (g.std() + 1e-9)
    w = np.exp(z / tau)
    return w / w.sum()


def aggregate(embeddings: torch.Tensor, w: np.ndarray) -> torch.Tensor:
    """Weighted, L2-normalised aggregation of sentence embeddings."""
    w = torch.as_tensor(w, dtype=embeddings.dtype, device=embeddings.device).unsqueeze(1)
    return F.normalize((embeddings * w).sum(0), dim=0)
