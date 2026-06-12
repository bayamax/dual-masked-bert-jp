"""A/B eval for the conditioned-confidence router (intent_route.py) vs the shipping
`intent_of` decision rule (accept top-1 iff p>=0.5, else full regex fallback).

Protocol: rebuild the training corpus exactly as evals/intent_train.py does (hand-labelled
core + intent_gen.jsonl), embed with the same BGE-small, then 5-fold stratified CV. Inside
each fold the LogisticRegression is fit on the train split only and BOTH decision rules
route the held-out split, so the comparison is honest end-to-end (not just clf accuracy).

Also scores a hand-written "Kyoto" challenge set (the STATUS 2026-06-10 failure mode:
low-confidence questions with an incidental I/my that the regex fallback yanks to recall)
using a model trained on the full corpus.

Run from anywhere:  python3 eval_intent_routing.py --hf-repo /path/to/hypernet-sp-distill
"""
import argparse, importlib.util, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intent_route import route_intent, regex_intent, is_question, looks_mathy


def load_corpus(hf):
    spec = importlib.util.spec_from_file_location("intent_train", os.path.join(hf, "evals", "intent_train.py"))
    it = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(it)
    items = {k: list(v) for k, v in it.DATA.items()}
    have = {t for v in items.values() for t in v}
    gen = os.path.join(hf, "evals", "intent_gen.jsonl")
    if os.path.exists(gen):
        for l in open(gen):
            o = json.loads(l)
            if o["label"] in items and o["text"] not in have:
                items[o["label"]].append(o["text"]); have.add(o["text"])
    texts, y = [], []
    for lab, ts in items.items():
        texts += ts; y += [lab] * len(ts)
    return texts, np.array(y)


def embed(hf, texts, cache="bge_cache.npz"):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if list(d["texts"]) == texts:
            return d["X"]
    sys.path.insert(0, os.path.join(hf, "runtime"))
    from rag import BGERetriever
    bge = BGERetriever()
    X = np.concatenate([bge._encode(texts[i:i + 64], is_query=False) for i in range(0, len(texts), 64)], 0)
    np.savez(cache, X=X, texts=np.array(texts, dtype=object))
    return X


class _Probe:                                     # adapts a fit clf + precomputed vector to route_intent
    def __init__(self, vec):
        self.vec = vec

    def _encode(self, texts, is_query=False):
        return self.vec[None]


def baseline_route(text, clf, x, hi=0.5):
    """The shipping rule: top-1 iff p>=hi (with the interrogative-fact fix), else full regex."""
    p = clf.predict_proba([x])[0]
    i = int(p.argmax())
    if p[i] >= hi:
        lab = str(clf.classes_[i])
        if lab == "fact" and is_question(text):
            lab = "math" if looks_mathy(text) else "recall"
        return lab
    return regex_intent(text)


# STATUS failure mode + neighbours: questions whose embedding lands near 0.5 and whose
# surface has an incidental personal pronoun. Gold labels follow the class definitions in
# evals/intent_train.py (recall = asks for the user's OWN stored info; lookup = world fact;
# chitchat = open-ended ideas/explanations).
CHALLENGE = [
    ("any tips for my weekend trip to Kyoto?", "chitchat"),
    ("what should we do on our weekend trip to Kyoto?", "chitchat"),
    ("can you suggest things for me to see in Kyoto?", "chitchat"),
    ("what's a good souvenir to bring back from my trip?", "chitchat"),
    ("how do I get from Kyoto station to Kinkaku-ji?", "lookup"),
    ("what's the weather like in Kyoto in November?", "lookup"),
    ("where did I park my car?", "recall"),
    ("what hotel did I say I'm staying at?", "recall"),
    ("remind me my flight time", "recall"),
    ("I'm off to Kyoto this weekend!", "fact"),
    ("if my hotel is $120 a night for 3 nights, what's the total?", "math"),
    ("what do you think I should cook tonight?", "chitchat"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default=".")
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--lo", type=float, default=0.30)
    a = ap.parse_args()

    texts, y = load_corpus(a.hf_repo)
    X = embed(a.hf_repo, texts)
    print(f"corpus: {len(y)} examples | {[(l, int((y == l).sum())) for l in sorted(set(y))]}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    base_hit = new_hit = 0
    grey = grey_base_hit = grey_new_hit = 0
    top2_cover = grey_n = 0
    flips = []                                     # (text, gold, base, new) where the rules disagree
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced").fit(X[tr], y[tr])
        bundle = {"clf": clf}
        for i in te:
            p = clf.predict_proba([X[i]])[0]
            p1 = float(p.max())
            order = np.argsort(-p)
            b = baseline_route(texts[i], clf, X[i], hi=a.hi)
            n = route_intent(texts[i], bundle, _Probe(X[i]), hi=a.hi, lo=a.lo)
            base_hit += b == y[i]; new_hit += n == y[i]
            if a.lo <= p1 < a.hi:
                grey_n += 1
                grey_base_hit += b == y[i]; grey_new_hit += n == y[i]
                top2_cover += y[i] in {str(clf.classes_[order[0]]), str(clf.classes_[order[1]])}
            if b != n:
                flips.append((texts[i], y[i], b, n, p1))
    N = len(y)
    print(f"\n== 5-fold CV end-to-end routing accuracy ==")
    print(f"  baseline (p>=0.5 else full regex): {base_hit}/{N} = {base_hit / N:.3f}")
    print(f"  conditioned (3-band, top-2 arb.):  {new_hit}/{N} = {new_hit / N:.3f}")
    if grey_n:
        print(f"\n== grey zone only ({a.lo} <= p1 < {a.hi}; {grey_n} examples = {grey_n / N:.1%} of corpus) ==")
        print(f"  gold label inside clf top-2: {top2_cover}/{grey_n} = {top2_cover / grey_n:.3f}")
        print(f"  baseline accuracy: {grey_base_hit}/{grey_n} = {grey_base_hit / grey_n:.3f}")
        print(f"  conditioned accuracy: {grey_new_hit}/{grey_n} = {grey_new_hit / grey_n:.3f}")
    print(f"\n== decisions that flipped ({len(flips)}) ==")
    for t, g, b, n, p1 in flips[:30]:
        print(f"  [{'NEW✓' if n == g else ('OLD✓' if b == g else 'both✗')}] p1={p1:.2f} gold={g:<8} "
              f"old={b:<8} new={n:<8} | {t[:60]}")

    # challenge set with a full-corpus model
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced").fit(X, y)
    bundle = {"clf": clf}
    sys.path.insert(0, os.path.join(a.hf_repo, "runtime"))
    from rag import BGERetriever
    bge = BGERetriever()
    print(f"\n== Kyoto challenge set (STATUS failure mode) ==")
    cb = cn = 0
    for t, gold in CHALLENGE:
        x = bge._encode([t], is_query=False)[0]
        p1 = float(clf.predict_proba([x])[0].max())
        b = baseline_route(t, clf, x, hi=a.hi)
        n = route_intent(t, bundle, bge, hi=a.hi, lo=a.lo)
        cb += b == gold; cn += n == gold
        print(f"  p1={p1:.2f} gold={gold:<8} old={b:<8}{'✓' if b == gold else '✗'} "
              f"new={n:<8}{'✓' if n == gold else '✗'} | {t}")
    print(f"  challenge: baseline {cb}/{len(CHALLENGE)}  conditioned {cn}/{len(CHALLENGE)}")
    print("INTENT_ROUTING_EVAL_DONE")


if __name__ == "__main__":
    main()
