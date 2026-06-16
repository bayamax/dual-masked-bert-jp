# STATUS 2026-06-16 — Faithful runtime: router removed, app-quality iteration (RunPod GPU)

We ran the CANONICAL on-device runtime (`hypernet_sp/app_session_torch.py`, the public
`baya1116/hypernet-sp-distill` repo) for real, conversed with it, fixed what broke, and iterated.
CPU was usable at first (~30–200 s/turn) then degraded after a container restart (>7 min/turn), so
we moved to a RunPod GPU pod (RTX A5000, ~$0.22/hr): ~1–3 s/turn, real iteration.

Reproduce: `faithful_setup.py` (rebuild from public HF) → `build_fft_hf.py` → `faithful_fixes.py`
(apply `faithful_patches/app_session_torch.patch`) → `conv_battery.py`. On GPU it's the one-shot
`boot_runpod.sh` (serves results on :8000 via the RunPod proxy).

## The big change the user asked for: REMOVE intent routing
The 6-way intent classifier (fact-ack / recall / lookup→web / command / math / chitchat) caused
the worst bugs (casual chat acked as a saved "fact"; "capital of France" refused offline because
lookups were web-only). It is GONE. Every turn now takes ONE generative path with GATED on-demand
retrieval (the "gated BGE recall + web" design we validated earlier):
- recall gate = BGE memory search, run BEFORE logging the turn (else it self-matches and echoes);
  chunks injected only above the relevance floor, and a low-confidence (<0.62) match is dropped
  rather than surfaced as a "closest note";
- web gate = a knowledge question with no memory hit + web available → web search;
- otherwise just generate → world questions answer from parametric knowledge ("Paris") instead
  of refusing offline.

## Fixes (all in faithful_patches/app_session_torch.patch; discovered by conversing)
| # | problem found in conversation | fix |
|---|---|---|
| 2 | math answers leaked `\boxed{1566.67}` | strip LaTeX in display normalization |
| 3 | bare 1.5B refused benign chitchat / degenerated | ambient persona ALWAYS in the MQ prefix |
| 4 | router mis-acked casual as facts; refused known Qs offline | remove router → unified gated path |
| 6 | open Qs hijacked by loose recall → "closest note" | drop <0.62 memory matches, just answer |
| 7 | arithmetic follow-up ("add 10 to that") ignored prior result | treat back-referencing arithmetic as compute |
| 8 | chitchat instruction-frame narrated verbatim + role-played user; `<\|end_of_thought\|>` leak | revert the user-turn frame; catch refusals in _ok → higher-temp retry; strip special tokens |

## End state (18-turn battery on GPU, all ~1–5 s)
Working: no refusals anywhere; chitchat replies; "capital of France"→Paris and other known facts
offline; 18×7→126 (fresh); follow-up "add 10 to that"→136; save→ack; recall after distraction
(JL412, 47000); open advice ("tips to relax") answered; "fun fact about the ocean" answered;
clean poem (no token leak).

## Remaining — now 1.5B-capability-bound, not routing/logic
- Standalone math is occasionally flaky once a lot of chitchat is in the SP context (one run
  degenerated "18×7" to meta-text). SP-vs-compute interference; would need compute-path isolation.
- Declarative statements get a slightly bland near-echo reply (engagement, not correctness) —
  the trade for removing the leaking instruction-frame.
- Factual confabulation ("Romeo and Juliet … adapted by Maria Roquett, 1607") — pure 1.5B content.

These are model-quality limits of the 1.5B distill, not the runtime logic.
