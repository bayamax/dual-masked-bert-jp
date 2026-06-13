"""recall_kit.gate — per-position recall GATE.

Decides, from the current decode-position hidden state alone, whether the answer needs a
fact that has scrolled out of the exposure window (>512 tokens back). Trained on real
Dolphin-R1 long-CoT attention (heldout AUC ~0.96). Pure-numpy at inference: no per-turn
state, only the live hidden query is consumed.
"""
import numpy as np

LAYERS = (8, 14, 20)


class RecallGate:
    def __init__(self, coef, intercept, mean, scale, thresh):
        self.coef = np.asarray(coef, np.float32).ravel()
        self.intercept = float(intercept)
        self.mean = np.asarray(mean, np.float32)
        self.scale = np.asarray(scale, np.float32)
        self.thresh = float(thresh)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["coef"], z["intercept"], z["mean"], z["scale"], z["thresh"])

    def score(self, q_flat):
        """q_flat: concatenated pre-RoPE q at LAYERS for the current position, shape [4608]."""
        x = (np.asarray(q_flat, np.float32).ravel() - self.mean) / self.scale
        return float(x @ self.coef + self.intercept)

    def fires(self, q_flat):
        return self.score(q_flat) > self.thresh
