"""Unit tests for decode_policy.py — pure stdlib:  python3 test_decode_policy.py"""
from decode_policy import DecodePolicy, canon_num

# clears min_think_chars without tripping the n-gram loop trigger (every word distinct)
PAD = " ".join(f"reasoning{i} step{i} detail{i}" for i in range(16)) + ". "


def check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    return bool(cond)


def main():
    r = []
    print("== canonicalisation ==")
    r.append(check("comma/$/decimals fold", canon_num("$1,157.00") == "1157" == canon_num("1157")))
    r.append(check("real decimal kept", canon_num("3.14") == repr(3.14)))

    print("== convergence fires on the 1157-style loop ==")
    p = DecodePolicy(k=3)
    t = PAD + "so the amount is 5 years... the total is 1157. "
    r.append(check("1 assertion: no fire", not p.note_text(t)))
    t += "Wait, let me recompute. 1000*1.03^5... so the answer is 1157. "
    r.append(check("2 assertions: no fire", not p.note_text(t)))
    t += "Checking again, that gives us 1157."
    r.append(check("3rd assertion fires", p.note_text(t)))
    r.append(check("converged value extracted", p.converged_answer() == "1157"))
    r.append(check("fires exactly once", not p.note_text(t + " and equals 1157 again")))

    print("== non-convergence does NOT fire ==")
    p = DecodePolicy(k=3)
    t = PAD + "the answer is 10. Hmm wait, actually equals 12. No — the total is 9."
    r.append(check("3 different values: no fire", not p.note_text(t)))
    p = DecodePolicy(k=3)
    r.append(check("operands alone don't count",
                   not p.note_text(PAD + "we have 5 boxes and 5 pencils and 5 days here")))
    p = DecodePolicy(k=3)
    r.append(check("short preamble can't fire", not p.note_text("the answer is 4. is 4. is 4.")))

    print("== boxed counts toward convergence ==")
    p = DecodePolicy(k=2)
    r.append(check("boxed + assertion", p.note_text(PAD + "the answer is 42... \\boxed{42}")))

    print("== think-loop trigger (question-rereading rut, no asserted values) ==")
    p = DecodePolicy(k=3)
    rut = PAD + ('"How many clips did she sell altogether in April and May?" ' * 5)
    r.append(check("6-gram x4 rut fires", p.note_text(rut)))
    p = DecodePolicy(k=3)
    r.append(check("varied reasoning does not fire", not p.note_text(
        PAD + "First compute April: 48. May is half: 24. Sum both months. Check units. Verify once.")))

    print("== two-phase temperature ==")
    p = DecodePolicy()
    r.append(check("think keeps base temp", p.temp(True, 0.6) == 0.6))
    r.append(check("answer phase ~greedy", p.temp(False, 0.6) <= 1e-3))
    r.append(check("opt-out respected", DecodePolicy(greedy_after_think=False).temp(False, 0.6) == 0.6))

    print(f"\n{sum(r)}/{len(r)} passed")
    assert all(r), "decode_policy tests failed"
    print("TEST_DECODE_POLICY_DONE")


if __name__ == "__main__":
    main()
