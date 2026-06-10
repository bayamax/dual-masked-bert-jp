"""Decode policy for release: (1) two-phase temperature, (2) answer-oscillation early stop.

Motivation (APP_HANDOFF / RESULTS):
  * multi-turn consistency is "high-variance run-to-run — 2-4/5": the final answer is
    sampled at temp 0.6 like the reasoning, so a correct chain can still verbalise a wrong
    final token. Sampling diversity is only useful INSIDE <think>; after </think> we want
    the argmax. -> two-phase: temp 0.6 in think, greedy after.
  * "long-reasoning self-degradation — correct 1157 mid-think, wrong final 1000": the model
    keeps re-deriving the same value and eventually talks itself out of it. The per-turn
    token cap is shape-blind. -> watch candidate answers in the think stream; when the SAME
    canonical value has been produced `k` times, force `\n</think>\n\n` (s1-style budget
    forcing, but triggered by CONVERGENCE rather than length).

Pure python over the decoded text stream — no MLX/torch import, so the same module drops
into tiered_rag_mlx._gen_once (MLX) and any torch rollout.

Integration (tiered_rag_mlx._gen_once inner loop):
    pol = DecodePolicy()
    ...
    t = int(mx.random.categorical(last * (1.0 / pol.temp(in_think, temp or TEMP))).item())
    ...after appending the decoded piece:
    if in_think and pol.note_text(tok.decode(gen[think_start:])) and fi >= len(feed):
        feed += list(tok.encode("\n</think>\n\nFinal answer: ", add_special_tokens=False))
"""
import re

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
# a concluding value: number (with optional $/%/units glued on) on a line that asserts a
# result. We deliberately key on assertion verbs so intermediate operands don't count.
_ASSERT_NUM = re.compile(
    r"(?:answer is|answer:|equals|=|total (?:is|of)|result is|gives us|so it'?s|"
    r"that(?:'s| is)|therefore,?)\s*\$?(-?\d[\d,]*(?:\.\d+)?)", re.I)


def canon_num(s):
    """Canonical numeric form: strip $, commas, trailing zeros ('1,157.00' == '1157')."""
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return s


class DecodePolicy:
    """Streaming convergence detector + phase-aware temperature."""

    def __init__(self, k=3, greedy_after_think=True, min_think_chars=200):
        self.k = k                                   # same value asserted k times -> converged
        self.greedy_after_think = greedy_after_think
        self.min_think_chars = min_think_chars       # don't fire on a trivial first line
        self.counts = {}
        self.fired = False

    def temp(self, in_think, base_temp):
        """Phase temperature: diversity inside <think>, argmax for the user-facing answer."""
        if in_think or not self.greedy_after_think:
            return base_temp
        return 1e-4                                  # ~greedy without a divide-by-zero special case

    def note_text(self, think_text):
        """Feed the CURRENT think text (decoded so far). Returns True exactly once, when the
        same canonical asserted value has appeared >= k times -> caller force-closes think."""
        if self.fired or len(think_text) < self.min_think_chars:
            return False
        self.counts = {}
        for m in _ASSERT_NUM.finditer(think_text):
            c = canon_num(m.group(1))
            self.counts[c] = self.counts.get(c, 0) + 1
        for m in _BOXED.finditer(think_text):
            c = canon_num(m.group(1))
            self.counts[c] = self.counts.get(c, 0) + 1
        if self.counts and max(self.counts.values()) >= self.k:
            self.fired = True
            return True
        return False

    def converged_answer(self):
        """The value the think stream converged on (majority assertion), or None."""
        if not self.counts:
            return None
        return max(self.counts.items(), key=lambda kv: kv[1])[0]
