# GainNet: sentence-level neural gain from fMRI
weights： https://huggingface.co/CopTain/neuroGainnet

A public release of **2,700 small networks** that map an English sentence to a scalar
**neural gain** — an estimate of how strongly that sentence drives the response of a
given brain region, measured with fMRI in 30 participants viewing 2-second video clips.

One sentence in, one number out (per region).

```
"A man is holding a bow and arrow, preparing to shoot at a target."   ->  0.41
"The image also shows some trees in the background."                  ->  0.33
```

Numbers above are illustrative; run the code for actual values.

---

## What is in this repository

| Path | Description |
|---|---|
| `gainnet/` | Inference code (`GainScorer`) |
| `weights/gainnet_fp16.pt` | 2,700 networks, fp16 (~200 MB) — **hosted on Hugging Face**, see below |
| `weights/gains_sentence_region.npz` | Gain table: 66,270 sentences × 90 regions (participant-averaged) |
| `weights/sentences.csv.gz` | The 66,270 sentences with video id and onset time |
| `examples/` | Minimal usage examples |
| `scripts/export_release.py` | How the release artifacts were generated from the original checkpoints |
| `DATA.md` | Data provenance, preprocessing, train/test split |
| `MODEL_CARD.md` | Intended use, limitations, biases |

Large files are hosted on the Hugging Face Hub:

```bash
# git-lfs or the huggingface CLI
huggingface-cli download <org>/gainnet --local-dir weights/
```

## Install

```bash
git clone <this-repo>
cd gainnet-release
pip install -r requirements.txt
```

`torch` and `sentence-transformers` are the only non-trivial dependencies. The
sentence encoder is `sentence-transformers/all-MiniLM-L12-v2`, used **without**
normalisation (the networks were trained on unnormalised embeddings).

## Quickstart

```python
from gainnet import GainScorer

scorer = GainScorer("weights/gainnet_fp16.pt")     # CPU is fine

sentences = [
    "A man is holding a bow and arrow, preparing to shoot at a target.",
    "The image also shows some trees in the background.",
]
g = scorer.score(sentences)        # shape (2, 90) — per-region gains
g_mean = scorer.score_mean(sentences)   # shape (2,)  — averaged over regions
print(g_mean)
```

Use it as a soft weight over sentence embeddings:

```python
import torch
from gainnet.scorer import weights_for, aggregate

# embeddings: (n_sent, 384) sentence embeddings of one paragraph
w = weights_for(sentences, g_mean, tau=1.0)
paragraph_vector = aggregate(embeddings, w)
```

`tau` is the sharpness of the weighting: large `tau` approaches a uniform average,
small `tau` approaches picking the single highest-gain sentence. It should be
selected on your data; the paper's analyses scan it.

## What the number is, and is not

**Is:** a scalar per (sentence, region, participant), the output of a 384→100→10→1
sigmoid network trained to predict the regional BOLD response to that sentence.
Releasing the participant-averaged version (mean over 30 participants) is the default.

**Is not:** an attention weight, a saliency map over image pixels, a memory score,
or a measure of descriptive accuracy. Gains for accurate and inaccurate captions are
not distinguished by construction.

## Validated properties

These are the findings the release is based on; see the paper for details.

**Semantic validity.** Gain correlates with human concreteness ratings in 86 of 90
regions after FDR correction (mean r = 0.56 uncorrected; 0.32 after controlling for
serial position, word frequency and sentence length).

**Shared across regions.** Pairwise correlations between regional gain series average
0.817; the first principal component explains 82.2% of the variance.

**Functionally, not anatomically, organised.** Clustering regional semantic profiles
yields five groups with an adjusted Rand index of only 0.10 against AAL labels, but
each group corresponds to a known functional system (visual, sensorimotor-parietal,
frontal-language, subcortical-limbic, orbitofrontal-temporal).

**Useful as a weight.** Weighted aggregation of sentence CLIP embeddings improves
image retrieval (top-1, 294 paragraphs: 67.7% → 72.1%, McNemar p = 0.007; 297 Qwen2-VL
paragraphs: 74.7% → 77.4%, p = 0.039) and cosine similarity (relative +0.5–0.7%,
paired d = 0.32–0.60). The effect disappears when gains are permuted within a paragraph.

**Not explained by word norms.** Concreteness alone reproduces the similarity gain but
not the retrieval gain consistently; arousal alone does nothing. The full gain is the
only condition significant in both out-of-sample retrieval sets.

## Known limitations

- **One dataset, one stimulus class.** 2-second clips with one-sentence captions.
  Whether the signal transfers to longer narratives is untested.
- **Fixed sentence encoder.** The networks are defined on `all-MiniLM-L12-v2`
  embeddings; another encoder requires retraining.
- **The gain is trained to predict BOLD, not to be useful.** Downstream usefulness
  was demonstrated on a modest benchmark (294 and 297 paragraphs), not a standard
  retrieval benchmark.
- **Arousal does not work as a weight** even though it correlates with gain.
- **Component decomposition is unreliable.** Splitting gain into concreteness /
  arousal / residual components with word-level norms is confounded by tokenisation
  and out-of-vocabulary coverage. Treat such decompositions as exploratory.

## License

Code: MIT. Derived weights and gain tables: CC BY 4.0, subject to the terms of the
source dataset (see `DATA.md`). Not for clinical use.

## Citation

See `CITATION.cff`. If you use the weights, please cite both this release and the
source dataset.

---

## 中文摘要

本仓库发布 **2,700 个逐句、逐区的神经增益网络**（30 名被试 × 90 个 AAL 脑区）。
输入一句英文描述，输出该句在某个脑区的响应强度（0–1 之间的标量）。
增益已验证与人类具体度评分相关、跨脑区高度共享、按功能系统而非解剖分区组织，
并可作为句子表征的权重改善图文检索。已知局限见 `MODEL_CARD.md`。
