"""Unit tests for anaphora.py — pure stdlib:  python3 test_anaphora.py"""
from anaphora import expand_web_query, needs_resolution, has_own_anchor


def check(name, got, want):
    ok = got == want
    print(f"  {'OK ' if ok else 'FAIL'} {name}: got={got!r}" + ("" if ok else f" want={want!r}"))
    return ok


def main():
    r = []
    pins = ["I'm planning a weekend trip to Kyoto."]                  # most-recent-last

    print("== the STATUS failure: 'there' follow-up hits the web without its antecedent ==")
    r.append(check("temples-there", expand_web_query("what are the best temples to visit there?", pins),
                   "what are the best temples to visit there? (Kyoto)"))
    r.append(check("food-it", expand_web_query("what food is it famous for?", pins),
                   "what food is it famous for? (Kyoto)"))

    print("== suppression: a query with its own anchor must NOT be expanded ==")
    r.append(check("own-city", expand_web_query("what are the best temples in Osaka?", pins),
                   "what are the best temples in Osaka?"))
    r.append(check("own-code", expand_web_query("is flight QX7-2291 delayed? they said maybe", pins),
                   "is flight QX7-2291 delayed? they said maybe"))
    r.append(check("no-anaphor", expand_web_query("best ramen restaurants?", pins),
                   "best ramen restaurants?"))

    print("== recency: the most recently mentioned entity wins ==")
    pins2 = ["I'm planning a weekend trip to Kyoto.", "actually, change it to Kanazawa instead"]
    r.append(check("correction-wins", expand_web_query("how cold is it there in winter?", pins2),
                   "how cold is it there in winter? (Kanazawa, Kyoto)"))
    r.append(check("cap-1", expand_web_query("how cold is it there in winter?", pins2, cap=1),
                   "how cold is it there in winter? (Kanazawa)"))

    print("== session fallback + person anaphora ==")
    r.append(check("session-person",
                   expand_web_query("what company does she run?", [], session=["look up Lisa Su for me"]),
                   "what company does she run? (Lisa Su)"))
    r.append(check("empty-memory", expand_web_query("what are the best temples there?", [], session=[]),
                   "what are the best temples there?"))

    print("== guards ==")
    r.append(check("needs-resolution", needs_resolution("what are the best temples there?"), True))
    r.append(check("anchor-detected", has_own_anchor("temples in Osaka"), True))
    r.append(check("initial-cap-not-anchor", has_own_anchor("There are good temples?"), False))
    r.append(check("generic-words-skipped",
                   expand_web_query("is it open on Monday?", ["The deadline is Friday, please remember"]),
                   "is it open on Monday?"))

    print(f"\n{sum(r)}/{len(r)} passed")
    assert all(r), "some anaphora tests failed"
    print("TEST_ANAPHORA_DONE")


if __name__ == "__main__":
    main()
