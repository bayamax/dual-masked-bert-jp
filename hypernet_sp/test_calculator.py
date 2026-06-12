"""Unit tests for calculator.py — pure stdlib:  python3 test_calculator.py
The claim corpus is taken from REAL failure transcripts (v5 P1/P7, composite B.math)."""
from calculator import safe_eval, find_claims, repair_answer


def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name}" + (f"  | {info}" if info and not cond else ""))
    return bool(cond)


def main():
    r = []
    print("== safe_eval ==")
    r.append(check("compound interest", abs(safe_eval("2000 * 1.05^3") - 2315.25) < 1e-9))
    r.append(check("fractions", abs(safe_eval("1/6 + 1/4 - 1/12") - 1 / 3) < 1e-12))
    r.append(check("money/commas/unicode", abs(safe_eval("$1,200 × 0.30") - 360) < 1e-9))
    r.append(check("latex times", abs(safe_eval(r"2000 \times 1.1025") - 2205) < 1e-9))
    r.append(check("implicit mult", abs(safe_eval("2(3+4)") - 14) < 1e-9))
    for bad in ("__import__('os')", "x+1", "(1).__class__", "lambda: 1"):
        try:
            safe_eval(bad); ok = False
        except ValueError:
            ok = True
        r.append(check(f"rejects {bad[:20]!r}", ok))

    print("== find_claims on real failure text ==")
    p1 = ("After 2 years: 2000 * 1.1025 = 2205. After 3 years we compute "
          "2000 * 1.157625 = 121550.625 dollars.")
    claims = find_claims(p1)
    r.append(check("two claims found", len(claims) == 2, str(claims)))
    r.append(check("correct claim passes", claims[0][3] is True))
    r.append(check("wrong claim caught", claims[1][3] is False and abs(claims[1][2] - 2315.25) < 1e-6))
    budget = "The budget is 650 and you spend 200, so 650 - 200 = 210."
    c2 = find_claims(budget)
    r.append(check("650-200=210 caught", len(c2) == 1 and not c2[0][3] and c2[0][2] == 450))
    rounded = "Average speed: 250 / 6.5 = 38.46 km/h. Total 250 / 6.5 = 38 roughly."
    c3 = find_claims(rounded)
    r.append(check("rounded claims tolerated", all(ok for *_, ok in c3), str(c3)))

    print("== repair_answer ==")
    fixed, cor = repair_answer("210", full_body=budget)
    r.append(check("final value repaired from body claim", fixed == "450" and len(cor) == 1))
    fixed, cor = repair_answer("The total is $121,550.625.",
                               full_body=p1)
    r.append(check("compound-interest repair", "2315.25" in fixed, fixed))
    fixed, cor = repair_answer("The answer is 24.", full_body="6 * 4 = 24 muffins")
    r.append(check("correct answers untouched", fixed == "The answer is 24." and not cor))
    fixed, cor = repair_answer("Kinkaku-ji and Kiyomizu-dera.", full_body="no math here")
    r.append(check("non-arithmetic untouched", not cor))

    print(f"\n{sum(r)}/{len(r)} passed")
    assert all(r), "calculator tests failed"
    print("TEST_CALCULATOR_DONE")


if __name__ == "__main__":
    main()
