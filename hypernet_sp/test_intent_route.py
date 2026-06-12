"""Unit tests for intent_route.py — no BGE/sklearn needed (the classifier is mocked),
so this runs in CI and on-device alike:  python3 test_intent_route.py"""
import numpy as np
import intent_route as ir


class FakeClf:
    classes_ = np.array(["chitchat", "command", "fact", "lookup", "math", "recall"])

    def __init__(self, probs):
        self.probs = np.array(probs, dtype=float)

    def predict_proba(self, X):
        return self.probs[None]


class FakeBGE:
    def _encode(self, texts, is_query=False):
        return np.zeros((len(texts), 384), dtype=np.float32)


def probs(**kw):
    base = {c: 0.0 for c in FakeClf.classes_}
    base.update(kw)
    v = np.array([base[c] for c in FakeClf.classes_])
    return v / v.sum()


def route(text, **kw):
    return ir.route_intent(text, {"clf": FakeClf(probs(**kw))}, FakeBGE())


def check(name, got, want):
    ok = got == want
    print(f"  {'OK ' if ok else 'FAIL'} {name}: got={got} want={want}")
    return ok


def main():
    r = []
    print("== high-confidence band: classifier decides ==")
    r.append(check("confident recall", route("where did I park?", recall=0.9, lookup=0.1), "recall"))
    r.append(check("confident lookup", route("how tall is Mount Fuji?", lookup=0.9, recall=0.1), "lookup"))

    print("== interrogative-fact fix routes by cue, not blanket recall ==")
    r.append(check("fact+directions -> lookup",
                   route("how do I get from Kyoto station to Kinkaku-ji?", fact=0.6, lookup=0.3, recall=0.1), "lookup"))
    r.append(check("fact+memory-cue -> recall",
                   route("what was my room number again?", fact=0.6, recall=0.3, lookup=0.1), "recall"))
    r.append(check("fact+mathy -> math",
                   route("how much is 17 times 24?", fact=0.6, math=0.3, lookup=0.1), "math"))
    r.append(check("fact+advice -> chitchat",
                   route("any tips for my weekend trip to Kyoto?", fact=0.6, lookup=0.3, chitchat=0.1), "chitchat"))

    print("== grey zone (0.30 <= p1 < 0.5): arbitrate within top-2 ==")
    r.append(check("advice question beats recall",
                   route("what do you think I should cook tonight?", recall=0.38, chitchat=0.35, lookup=0.27), "chitchat"))
    r.append(check("memory cue beats lookup",
                   route("what dosage did you mention?", lookup=0.4, recall=0.35, chitchat=0.25), "recall"))
    r.append(check("bare arithmetic beats chitchat",
                   route("add seven and thirteen", chitchat=0.4, math=0.35, lookup=0.25), "math"))
    r.append(check("no-evidence tie keeps top-1",
                   route("tell me about the stock market briefly", lookup=0.4, recall=0.35, chitchat=0.25), "lookup"))
    r.append(check("ideas question neither candidate claims -> chitchat",
                   route("what's a good souvenir to bring back from my trip?", recall=0.32, fact=0.26,
                         lookup=0.22, chitchat=0.2), "chitchat"))

    print("== creative override defers to math cues ==")
    r.append(check("write-out-calculation stays math",
                   route("Write out the calculation of 17 * 24.", math=0.4, chitchat=0.35,
                         fact=0.25), "math"))
    r.append(check("plain writing request stays chitchat",
                   route("Write a short poem about rain.", fact=0.6, chitchat=0.3,
                         lookup=0.1), "chitchat"))

    print("== low confidence / no model: regex fallback ==")
    r.append(check("regex fallback on p1<lo",
                   route("what's my insurance policy number?", recall=0.2, lookup=0.18, fact=0.17,
                         chitchat=0.16, math=0.15, command=0.14), "recall"))
    r.append(check("no model -> regex", ir.route_intent("what's 15% of 240", None, None), "math"))

    print(f"\n{sum(r)}/{len(r)} passed")
    assert all(r), "some intent_route tests failed"
    print("TEST_INTENT_ROUTE_DONE")


if __name__ == "__main__":
    main()
