"""Export the sklearn LogisticRegression heads (intent / specificity / attn-trigger) to
plain .npz + a numpy-only scorer — a release requirement, not a nicety:

  * the shipped .joblib files are version-locked pickles (loading the 1.7.2 artifacts under
    1.9.0 already warns InconsistentVersionWarning); an app update that bumps sklearn can
    silently change or break routing;
  * the iOS port (mlx-swift) has no sklearn at all — coef/intercept matrices are all it
    can consume.

`LinearHead` reproduces predict_proba bit-for-bit (softmax over W x + b for multinomial,
sigmoid for binary) and is verified against sklearn for every exported head.

Run next to evals/*.joblib and attn_trigger3.joblib:  python3 export_classifiers.py
"""
import os
import numpy as np


class LinearHead:
    """numpy-only stand-in for a fitted sklearn LogisticRegression (+ optional input
    standardisation, as used by the attention trigger)."""

    def __init__(self, coef, intercept, classes, mu=None, sd=None):
        self.coef, self.intercept = np.asarray(coef), np.asarray(intercept)
        self.classes = list(classes)
        self.mu = None if mu is None else np.asarray(mu)
        self.sd = None if sd is None else np.asarray(sd)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        return cls(d["coef"], d["intercept"], list(d["classes"]),
                   d["mu"] if "mu" in d else None, d["sd"] if "sd" in d else None)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.mu is not None:
            X = (X - self.mu) / self.sd
        z = X @ self.coef.T + self.intercept
        if z.shape[1] == 1:                                    # binary: sigmoid on the margin
            p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
            return np.stack([1.0 - p1, p1], 1)
        z = z - z.max(1, keepdims=True)                        # multinomial: softmax
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, X):
        return [self.classes[i] for i in self.predict_proba(X).argmax(1)]


def export(joblib_path, out_path, mu=None, sd=None):
    import joblib
    b = joblib.load(joblib_path)
    clf = b["clf"]
    np.savez(out_path, coef=clf.coef_, intercept=clf.intercept_,
             classes=np.array([str(c) for c in clf.classes_], dtype=object),
             **({} if mu is None else {"mu": mu, "sd": sd}))
    head = LinearHead.load(out_path)
    X = np.random.default_rng(0).normal(size=(64, clf.coef_.shape[1]))
    if mu is not None:
        ref = clf.predict_proba((X - mu) / sd)
    else:
        ref = clf.predict_proba(X)
    err = float(np.abs(head.predict_proba(X) - ref).max())
    print(f"  {joblib_path} -> {out_path}  classes={len(head.classes)} "
          f"dim={clf.coef_.shape[1]}  max|Δp| vs sklearn = {err:.2e}")
    assert err < 1e-9, "exported head does not reproduce sklearn probabilities"
    return head


def main():
    import joblib
    print("exporting classifier heads to portable npz:")
    if os.path.exists("evals/intent_clf.joblib"):
        export("evals/intent_clf.joblib", "evals/intent_head.npz")
    if os.path.exists("evals/specificity_clf.joblib"):
        export("evals/specificity_clf.joblib", "evals/specificity_head.npz")
    if os.path.exists("attn_trigger3.joblib"):
        b = joblib.load("attn_trigger3.joblib")
        np.savez("attn_trigger3_head.npz", coef=b["clf"].coef_, intercept=b["clf"].intercept_,
                 classes=np.array(["ctrl", "ref"], dtype=object), mu=b["mu"], sd=b["sd"],
                 heads=np.array(b["heads"]))
        h = LinearHead.load("attn_trigger3_head.npz")
        X = np.random.default_rng(1).normal(size=(32, len(b["heads"])))
        ref = b["clf"].predict_proba((X - b["mu"]) / b["sd"])
        err = float(np.abs(h.predict_proba(X) - ref).max())
        print(f"  attn_trigger3.joblib -> attn_trigger3_head.npz  heads={b['heads']}  max|Δp| = {err:.2e}")
        assert err < 1e-9
    print("EXPORT_CLASSIFIERS_DONE")


if __name__ == "__main__":
    main()
