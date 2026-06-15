"""faithful_fixes.py — app-quality fixes to the canonical runtime, applied to faithful_run/
after faithful_setup.py. Each fix is an idempotent string replacement on the upstream file
(faithful_run is gitignored; this script is the tracked record of the changes + rationale).

Run:  python3 faithful_setup.py && python3 faithful_fixes.py
Discovered by conversing with the live AppSession.turn() (see conv1.log transcripts).
"""
import os, sys
RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faithful_run")
APP = os.path.join(RUN, "app_session_torch.py")


def patch(path, old, new, tag):
    s = open(path).read()
    if new in s:
        print(f"[skip] {tag}: already applied"); return
    if old not in s:
        print(f"[WARN] {tag}: anchor not found — upstream changed?"); return
    open(path, "w").write(s.replace(old, new, 1))
    print(f"[ok]   {tag}")


# ---- FIX #1: casual chat misrouted as "fact" -> canned "Got it — saved." instead of a reply.
# The intent clf over-fires "fact" on first-person narrative ("work has been hectic", "the
# weather's nice", "tried a ramen place"). Only ack-save when there is something to save
# (explicit persist request or a specificity-probe value); else fall through to chitchat.
FIX1_OLD = '''        if intent == "fact":
            # facts bypass generation (OPERATING.md): log + instant ack. Generating here
            # wastes a full turn AND pollutes the conversation stream — the composite
            # battery showed the previous fact's ramble bleeding into the next chitchat.
            if store == "persist" or mc.wants_persist(user_msg):
                self.mem.persist(user_msg)
            else:
                self.mem.remember_session(user_msg)
            self._stitch(user_msg, "Got it — saved.")
            return "Got it — saved.", None, []'''
FIX1_NEW = '''        if intent == "fact":
            # facts bypass generation (OPERATING.md): log + instant ack. BUT the intent clf
            # over-fires "fact" on casual first-person narrative ("work has been hectic",
            # "the weather's been nice", "tried a new ramen place") -> every chitchat turn
            # got a canned "Got it — saved." instead of a reply. Only treat it as a savable
            # fact when there is something to save: an explicit persist request or a
            # specificity-probe value (JL412, 47000, a proper noun). Otherwise fall through
            # to a normal chitchat reply (still logged to session for recall context).
            if store == "persist" or mc.wants_persist(user_msg) or self.specific_spans(user_msg):
                if store == "persist" or mc.wants_persist(user_msg):
                    self.mem.persist(user_msg)
                else:
                    self.mem.remember_session(user_msg)
                self._stitch(user_msg, "Got it — saved.")
                return "Got it — saved.", None, []
            self.mem.remember_session(user_msg)
            intent = "chitchat"'''


def main():
    if not os.path.exists(APP):
        print("faithful_run/app_session_torch.py missing — run faithful_setup.py first"); sys.exit(1)
    patch(APP, FIX1_OLD, FIX1_NEW, "FIX1 casual-not-fact")


if __name__ == "__main__":
    main()
