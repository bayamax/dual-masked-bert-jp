# STATUS 2026-06-15 — Composite under the SP regime: recall + web coexist (CPU validation)

Question: can recall and web triggers coexist, evaluated on real tasks (GSM8K, Dolphin-R1,
multi-turn chat), with sensible behavior everywhere? Validated on the PUBLIC base model
(`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, CPU, no secrets) — the faithful fft-student /
real-SP version belongs on the GPU pod, but every number below is reproducible without secrets.

## Key correction
Generation always runs under SP compression. While doing CoT the raw window fills with reasoning
and the original problem (operands) is **evicted** -> SP loses high-entropy specifics. So for any
CoT task (GSM8K, Dolphin) **recall is MANDATORY, not a false-fire**. cotscan already showed
OFF (no recall) = 0/25 on GSM8K; recall restores it.

## Clarified design (why coexistence is natural)
- **WEB** is decided at the **question onset** (is this a knowledge gap?). Fire -> inject external doc.
- **RECALL** is decided **mid-CoT** (did SP eviction drop an operand?). Fire -> re-inject the block.
- Different times, different signals -> **no competition**. Run both independently, inject what fires.
  The only coexistence tuning is **web precision**: web must stay silent on reasoning/chat.

## Results
### 1. Recall is necessary under eviction — `sp_eval.py` (2-phase window-eviction proxy)
GSM8K, all 5 items evicted: **OFF acc 0.4 vs PIN (re-inject problem) acc 0.6**. Directionally
confirms recall helps under SP (q0 flips 2->18). Weak on CPU/N=5; the strong faithful result
(OFF 0/25 -> 36%) is the repo's pod run.

### 2. Web precision with reasoning as negatives — `web_precision_sp.py` (no generation, fast)
Web gate trained with GSM8K + Dolphin + known + casual as NEGATIVES, fabricated entities as
positives; threshold = 97th pct of held negatives:

| suite | fire | score band | want |
|---|---|---|---|
| web (knowledge gap) | **8/8** | [0.2, 4.3] | high |
| gsm8k | **0/10** | [-6.6, -5.4] | 0 |
| dolphin | **0/3** | [-6.0, -5.4] | 0 |
| casual | 0/5 | [-4.2, -2.8] | 0 |
| known | 1/6 | [-2.7, -0.5] | 0 (minor residual) |

Reasoning tasks sit far below threshold -> the web gate is strongly confident they are not gaps.
**Web stays silent on GSM8K/Dolphin** => safe to coexist with mid-CoT recall.

### 3. Compositional coexistence — `coexist_demo.py`
On BOTH turns (need an evicted fact AND a fabricated-entity fact) both gates fire, both
injections land, both golds recovered 3/3. Spurious fires are harmless because injection is gated
on retrievability (recall over-fire with no evicted block = no-op).

### 4. Web inject end-task — `web_path_test.py`
Knowledge-gap questions: no-inject ~0 (model can't know fabricated entities), inject restores the
answer. OOF AUC 0.93; at a zero-known-false-fire threshold web recall ~44% (precision/recall tunable).

## Bottom line
- Operational coexistence works: independent gates, inject-what-fires, retrievability-gated.
- Tuned so all suites are sensible: recall fires on reasoning (helps), web silent on reasoning
  (0/10 GSM8K, 0/3 Dolphin), web fires on gaps (8/8), chat quiet.
- Residual: known 1/6 web false-fire (raise threshold slightly) — calibration, not architecture.

## Next (faithful, GPU pod)
Reproduce on the fft student under real SP (pooler + block_recall) via the boot-script workflow:
mid-CoT per-position recall (pin-once, the stable recipe) + onset web gate, on full GSM8K/Dolphin
+ multi-turn needles. Scripts here are the CPU-validated blueprint.
