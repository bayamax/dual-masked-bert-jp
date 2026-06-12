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
# a concluding value: number (with optional $/%/units glued on) after an assertion verb.
# The verb list must cover ANSWER-shaped phrasings broadly: the composite battery showed a
# change-making chain where "she gets $26 back" went UNCOUNTED while "total is 24" was
# counted 3x — the detector then converged on the intermediate 24. Operand mentions
# (bare "50 - 24") still don't count.
_ASSERT_NUM = re.compile(
    r"(?:answer is|answer:|equals|=|total (?:is|of)|result is|gives us|so it'?s|"
    r"that(?:'s| is)|therefore,?|change (?:is|of|would be)|receives?|left with|"
    r"expecting)\s*\$?(-?\d[\d,]*(?:\.\d+)?)"
    r"|(?:gets?|got|give[sn]?|hand(?:ed)?)\s+\$?(-?\d[\d,]*(?:\.\d+)?)\s+(?:back|in change)",
    re.I)


def _assert_values(text):
    """Canonical values of all answer-shaped assertions, in order."""
    return [canon_num(m.group(1) or m.group(2)) for m in _ASSERT_NUM.finditer(text)]


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

    LOOP_NGRAM, LOOP_REPS = 6, 4                     # a 6-gram seen 4x in think = a rut

    def _think_loop(self, text):
        """Second trigger (smoke finding): over-thinking that never ASSERTS a value — e.g.
        re-reading the question over and over — evades the convergence counter. Catch it as
        n-gram repetition over the think stream (same idea as the answer-side degeneration
        guard, longer n / higher bar so legitimate re-derivation doesn't fire)."""
        w = text.split()
        if len(w) < self.LOOP_NGRAM * self.LOOP_REPS:
            return False
        from collections import Counter
        tri = Counter(" ".join(w[i:i + self.LOOP_NGRAM]) for i in range(len(w) - self.LOOP_NGRAM + 1))
        return tri.most_common(1)[0][1] >= self.LOOP_REPS

    def note_text(self, think_text):
        """Feed the CURRENT think text (decoded so far). Returns True exactly once, when the
        same canonical asserted value has appeared >= k times (convergence) OR the think
        stream is stuck in an n-gram loop -> caller force-closes think."""
        if self.fired or len(think_text) < self.min_think_chars:
            return False
        vals = _assert_values(think_text) + [canon_num(m.group(1))
                                             for m in _BOXED.finditer(think_text)]
        self.counts = {}
        for c in vals:
            self.counts[c] = self.counts.get(c, 0) + 1
        if self.counts:
            mode, n_mode = max(self.counts.items(), key=lambda kv: kv[1])
            runner_up = max((v for c, v in self.counts.items() if c != mode), default=0)
            asserted = _assert_values(think_text)
            last = asserted[-1] if asserted else None
            # converged = the mode is asserted k+ times, is ALSO the latest assertion, and
            # has no live competitor (a value asserted 2+ times means the chain is still
            # weighing two candidates — forcing now would pick a side arbitrarily; the
            # composite battery's change-making chain is exactly this shape).
            if n_mode >= self.k and last == mode and runner_up < 2:
                self.fired = True
                return True
        if self._think_loop(think_text):
            self.fired = True
            return True
        return False

    def converged_answer(self):
        """The value the think stream converged on (majority assertion), or None."""
        if not self.counts:
            return None
        return max(self.counts.items(), key=lambda kv: kv[1])[0]
