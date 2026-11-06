"""Minimal usage: score sentences and use the gains as weights.

Run from the repository root:

    python examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gainnet import GainScorer, REGIONS

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "weights", "gainnet_fp16.pt")


def main():
    scorer = GainScorer(WEIGHTS)

    sentences = [
        "A man is holding a bow and arrow, preparing to shoot at a target.",
        "The image also shows some trees in the background.",
        "A woman is riding a horse across a field.",
    ]

    gains = scorer.score(sentences)              # (n_sent, 90)
    print(f"per-region gains: {gains.shape}")
    print(f"mean over regions: {np.round(gains.mean(1), 4)}")

    # per-region ranking for the first sentence
    order = np.argsort(-gains[0])[:5]
    print("\nTop 5 regions for sentence 1:")
    for k in order:
        print(f"  {REGIONS[k]:<12} {gains[0, k]:.3f}")

    # attach gains to the paragraph and inspect the weights
    from gainnet.scorer import weights_for

    para = sentences
    w = weights_for(para, scorer.score_mean(para), tau=1.0)
    print("\nweights within one paragraph:", np.round(w, 3))
    print("sum:", w.sum().round(6))


if __name__ == "__main__":
    main()
