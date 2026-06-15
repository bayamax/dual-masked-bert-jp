"""faithful_fixes.py — app-quality fixes to the canonical runtime, applied to faithful_run/
after faithful_setup.py. The fixes were discovered by conversing with the live AppSession.turn()
(see conv*.log transcripts) and are captured as a unified diff in faithful_patches/ so the whole
set reproduces from the upstream file in one apply. faithful_run is gitignored; this script +
the patch are the tracked record.

Run:  python3 faithful_setup.py && python3 faithful_fixes.py

Changes in faithful_patches/app_session_torch.patch:
  FIX3  ambient assistant persona ALWAYS in the MQ prefix (even with no saved facts) — the bare
        1.5B refused benign chitchat ("I just adopted two kittens" -> "I can't assist") and
        emitted garbage name lists; persona keeps chitchat warm/on-task.
  FIX2  strip raw LaTeX from user-facing answers (\\boxed{1566.67} -> 1566.67).
  FIX4  REMOVE the 6-way intent router (fact-ack / recall / lookup->web / command / math /
        chitchat). Unify to one generative turn with GATED on-demand retrieval:
          - recall gate: BGE-search memory on QUESTIONS only (relevance floor = the gate),
            retrieved BEFORE logging the turn (else it self-matches and echoes the user);
          - web gate: a knowledge question with no memory hit + web available -> web search;
          - otherwise just generate (so "capital of France" is answered from parametric
            knowledge instead of refusing offline);
          - compute context is pulled only when the math turn back-references it ("add 10 to
            that"), so a self-contained "18 times 7" doesn't drag in unrelated pins.
        (This subsumes the earlier casual-not-fact fix: casual statements now generate.)
"""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "faithful_run")
APP = os.path.join(RUN, "app_session_torch.py")
PATCH = os.path.join(HERE, "faithful_patches", "app_session_torch.patch")


def main():
    if not os.path.exists(APP):
        print("faithful_run/app_session_torch.py missing — run faithful_setup.py first"); sys.exit(1)
    # idempotent: --forward skips hunks already applied; check reverse-apply to detect "done".
    done = subprocess.run(["patch", "-R", "-s", "-f", "--dry-run", APP, PATCH],
                          capture_output=True, text=True).returncode == 0
    if done:
        print("[skip] app_session_torch.py already patched"); return
    r = subprocess.run(["patch", "-s", "--forward", APP, PATCH], capture_output=True, text=True)
    if r.returncode == 0:
        print("[ok]   applied faithful_patches/app_session_torch.patch")
    else:
        print("[FAIL] patch did not apply cleanly — upstream may have changed:")
        print(r.stdout, r.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
