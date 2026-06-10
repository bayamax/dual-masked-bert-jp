"""Torch port of the FULL app turn pipeline (tiered_rag_mlx.ChatSession.turn) so the
composite app-usage battery can run off-Mac. Mirrors the MLX runtime decision-for-decision:

  intent routing (intent_route + real intent_clf)  ->  specificity pinning (real probe)
  -> tiered retrieval (memory_core.TieredMemory + BGE; web backend injected)
  -> context-injection prompts (verbatim aug templates from tiered_rag_mlx)
  -> bounded SP-evict generation (pooler + fft_hf) with DecodePolicy
  -> recall via isolated _clean_quote  ->  groundedness gate + rolled-back retries
  -> memory writes (persist / session / fact-only)

Differences from the MLX runtime are confined to: torch tensors, smaller default budgets
(CPU), and the injected `web` object (tests use a canned corpus; production uses
DuckDuckGo/Wikipedia exactly as before).
"""
import os, re, sys, time
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memory_core as mc
from intent_route import route_intent, looks_mathy
from anaphora import expand_web_query
from web_guard import guard_chunks
from decode_policy import DecodePolicy

ANSCAP = 600

# cheap candidate enumerator for the specificity probe (verbatim from tiered_rag_mlx)
_CAND = re.compile(
    r"\$\d[\d,]*(?:\.\d+)?"
    r"|\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b"
    r"|\b\d+(?:\.\d+)?\s?(?:cm|mm|km|kg|%|percent|days?|years?|hours?|min)\b"
    r"|\b[A-Za-z]*\d[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b"
    r"|\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b", 0)


class AppSession:
    def __init__(self, llm, tok, pooler, bge, intent_clf, spec_clf, mem, web=None,
                 rw=512, C=64, cap=600, temp=0.6, maxD=4096, seed=0):
        self.llm, self.tok, self.pooler, self.bge = llm, tok, pooler, bge
        self.intent_clf, self.spec_clf = intent_clf, spec_clf
        self.mem, self.web = mem, web
        self.rw, self.C, self.cap, self.temp, self.maxD = rw, C, cap, temp, maxD
        self.embT = llm.get_input_embeddings()
        self.eos = tok.eos_token_id
        self.THINK_OPEN = tok.encode("<think>\n", add_special_tokens=False)
        self.gen, self.kept, self.absorbed = [], [], 0
        self.evictions = 0
        torch.manual_seed(seed)

    def _evict(self, kept):
        """Mass-based eviction (port of sp_mlx.evict / MAXD): when the distant buffer
        exceeds maxD, keep the maxD tokens the pooler itself attends to most, in
        chronological order. The production path RESULTS.md flags as barely exercised —
        the 6k/12k long-haul battery exists to finally hit it."""
        if not self.maxD or len(kept) <= self.maxD:
            return kept
        _, mass = self.pooler.forward_with_mass(self._emb(kept).float())
        idx = sorted(mass[0].topk(self.maxD).indices.tolist())
        self.evictions += 1
        return [kept[i] for i in idx]

    # ---- learned heads -------------------------------------------------------------
    def intent_of(self, text):
        return route_intent(text, self.intent_clf, self.bge)

    def specific_spans(self, text, min_p=0.6, cap=6):
        if not self.spec_clf or not self.bge:
            return []
        cands = [c for c in dict.fromkeys(m.group(0).strip() for m in _CAND.finditer(text))
                 if len(c) >= 2][:24]
        if not cands:
            return []
        X = self.bge._encode(cands, is_query=False)
        si = list(self.spec_clf["clf"].classes_).index(1)
        p = self.spec_clf["clf"].predict_proba(X)[:, si]
        hits = sorted(((c, float(pp)) for c, pp in zip(cands, p) if pp >= min_p),
                      key=lambda h: -h[1])
        return [c for c, _ in hits[:cap]]

    # ---- generation core (port of sp_mlx/_gen_once bounded loop) --------------------
    def _emb(self, ids):
        return self.embT(torch.tensor([ids])) if ids else \
            torch.zeros(1, 0, self.pooler.H, dtype=self.embT.weight.dtype)

    @torch.no_grad()
    def _gen_once(self, aug, policy=None, cap=None):
        cap = cap or self.cap
        tok, llm = self.tok, self.llm
        gen, kept, absorbed = self.gen, self.kept, self.absorbed
        feed = list(tok.encode(("<｜end▁of▁sentence｜>" if gen else "") +
                               f"<｜User｜>{aug}<｜Assistant｜>", add_special_tokens=False)) \
            + list(self.THINK_OPEN)
        start, fi, new = len(gen), 0, 0
        cache = DynamicCache()
        llm(input_ids=torch.tensor([tok.encode("")]) if tok.encode("") else
            torch.tensor([[tok.bos_token_id]]), past_key_values=cache, use_cache=True)
        MQ = cache.get_seq_length()
        in_think, forced_final, done = True, False, False
        policy = policy or DecodePolicy()
        while not done:
            c0 = len(gen); R = min(c0, self.rw); nd_end = c0 - R
            if nd_end > absorbed:
                kept.extend(gen[absorbed:nd_end]); absorbed = nd_end
                kept = self._evict(kept)
            sp = self.pooler(self._emb(kept).float()).to(self.embT.weight.dtype)
            parts = [sp] + ([self._emb(gen[c0 - R:c0])] if R > 0 else [])
            block = torch.cat(parts, 1)
            cache.crop(MQ)
            last = llm(inputs_embeds=block, past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
            for _ in range(self.C):
                if fi < len(feed):
                    t = feed[fi]; fi += 1
                else:
                    T = policy.temp(in_think, self.temp)
                    t = int(torch.multinomial(F.softmax(last[0] / T, -1), 1))
                    if t == self.eos:
                        done = True; break
                    new += 1
                    if new >= cap:
                        done = True; break
                gen.append(t)
                if in_think and "</think>" in tok.decode(gen[-8:]):
                    in_think = False
                if in_think and fi >= len(feed) and policy.note_text(tok.decode(gen[start:])):
                    feed += list(tok.encode("\n</think>\n\nFinal answer: ",
                                            add_special_tokens=False))
                    in_think = False
                last = llm(inputs_embeds=self._emb([t]), past_key_values=cache,
                           use_cache=True).logits[:, -1, :].float()
            if done:
                break
        body = tok.decode(gen[start:]).split("<｜Assistant｜>", 1)[-1]
        if "</think>" not in body or mc._extract_answer(body) in mc._EMPTY:
            # pass-1 salvage (port of the MLX two-pass that this port was missing): the
            # think meandered to the cap without converging or looping — close it and force
            # a short greedy answer instead of returning the timeout token (v3 C7).
            def step(t):
                return llm(inputs_embeds=self._emb([t]), past_key_values=cache,
                           use_cache=True).logits[:, -1, :].float()
            for t in tok.encode("\n</think>\n\nFinal answer: ", add_special_tokens=False):
                gen.append(t); last = step(t)
            for _ in range(48):
                t = int(last[0].argmax())
                if t == self.eos:
                    break
                gen.append(t); last = step(t)
            body = tok.decode(gen[start:]).split("<｜Assistant｜>", 1)[-1]
        self.gen, self.kept, self.absorbed = gen, kept, absorbed
        ans = mc._extract_answer(body)
        if "Final answer:" in ans:
            ans = ans.split("Final answer:")[-1].strip()
        return ans[:ANSCAP]

    @torch.no_grad()
    def _clean_quote(self, aug, temp=0.2, think_budget=180, ans_budget=60):
        """Isolated recall: fresh cache, NO soft prompt, NO history (port of MLX version)."""
        tok, llm = self.tok, self.llm
        cache = DynamicCache()

        def feed(ids):
            return llm(inputs_embeds=self._emb(ids), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()

        def sample(last, budget):
            out, prev, rep = [], None, 0
            for _ in range(budget):
                t = int(torch.multinomial(F.softmax(last[0] / temp, -1), 1))
                if t == self.eos:
                    break
                rep = rep + 1 if t == prev else 0
                if rep >= 5:
                    break
                prev = t; out.append(t)
                last = llm(inputs_embeds=self._emb([t]), past_key_values=cache,
                           use_cache=True).logits[:, -1, :].float()
            return out
        last = feed(tok.encode(f"<｜User｜>{aug}<｜Assistant｜>", add_special_tokens=True)
                    + list(self.THINK_OPEN))
        think = sample(last, think_budget)
        last = feed(tok.encode("\n</think>\n\n", add_special_tokens=False))
        ans = tok.decode(sample(last, ans_budget)).strip()
        if mc._looks_degenerate(ans) or ans in mc._EMPTY:
            box = re.findall(r"\\boxed\{([^}]*)\}", tok.decode(think))
            ans = f"\\boxed{{{box[-1].strip()}}}" if box and box[-1].strip() else ans
        return ans[:ANSCAP]

    def _web_retrieve(self, query):
        if self.web is None:
            return None, []
        raw = self.web.search(query)
        ch = guard_chunks(raw[:2]) if raw else []
        return ("L3·web", ch) if ch else (None, [])

    # ---- the app turn (decision-for-decision port of ChatSession.turn) --------------
    def turn(self, user_msg, store="session", ack_only=False, retries=2):
        if ack_only:
            (self.mem.persist if store == "persist" else self.mem.remember_session)(user_msg)
            return "Got it — saved.", None, []
        intent = self.intent_of(user_msg)
        compute_like = intent in ("math", "command")
        # store-request phrased as a QUESTION ("Can you remember that my locker code is
        # 8042?"): interrogative shape routes it to recall, whose empty-retrieval honest
        # miss would answer "you haven't told me yet" to the very message telling us.
        # A persist verb + an assertable value = a save, whatever the punctuation.
        if mc.wants_persist(user_msg) and mc._FACTLIKE.search(user_msg) \
                and self.specific_spans(user_msg):
            self.mem.persist(user_msg)
            self.mem.pin(user_msg)
            return "Got it — saved.", None, []
        if intent not in ("recall", "lookup") and self.specific_spans(user_msg):
            self.mem.pin(user_msg)
        if intent == "fact":
            # facts bypass generation (OPERATING.md): log + instant ack. Generating here
            # wastes a full turn AND pollutes the conversation stream — the composite
            # battery showed the previous fact's ramble bleeding into the next chitchat.
            if store == "persist" or mc.wants_persist(user_msg):
                self.mem.persist(user_msg)
            else:
                self.mem.remember_session(user_msg)
            return "Got it — saved.", None, []
        if intent == "recall":
            src, chunks = self.mem.retrieve_personal(user_msg)
            if not chunks:
                # honest miss. Free generation here CONFABULATES (composite v2 B3: asked for
                # a never-stated wifi password, the model invented "password123"). A recall
                # is a lookup into the user's saved facts; an empty lookup has exactly one
                # truthful answer, and it costs zero tokens.
                return ("I don't have that saved — you haven't told me yet.", None, [])
        elif intent == "lookup":
            # known-fact first (strict): a personal question that surface-classifies as a
            # world lookup ('Where does my sister live?', 'Where does Daniel live?') must
            # quote what the user told us, not hit the web (composite v3 B3/B4).
            src, chunks = self.mem.retrieve_known(user_msg)
            if not chunks:
                wm_only = [p for p in self.mem.pins if p not in self.mem.session]
                mp = mc._sem_matches(user_msg, wm_only, self.bge, min_sim=0.5) if wm_only else None
                if mp:
                    src, chunks = "WM·pins", mp
                else:
                    q = expand_web_query(user_msg, self.mem.pins, self.mem.session)
                    src, chunks = self._web_retrieve(q)
        elif intent == "command" and (self.mem.session or self.mem.pins):
            log = self.mem.session[-self.mem.LOGCAP:]
            # a pin is redundant when a log line EXTENDS it ('<question> — result: 24'
            # startswith '<question>'): re-injecting the bare question next to its answered
            # form is what tangled the follow-up math turn in the composite battery.
            chunks = log + [c for c in self.mem.pins if not any(l.startswith(c) for l in log)]
            src = "L1·same-session" + ("+WM·pins" if self.mem.pins else "")
        elif intent == "math" and any(p != user_msg for p in self.mem.pins):
            prev = [p for p in self.mem.pins if p != user_msg and not mc._is_question(p)]
            log = [l for l in self.mem.session[-self.mem.LOGCAP:] if not mc._is_question(l)]
            chunks = log + [c for c in prev if not any(l.startswith(c) for l in log)]
            src = "WM·pins" + ("+L1" if log else "")
        else:
            src, chunks = None, []
        if chunks and compute_like:
            # keep compute injections SHORT and question-first. The verbatim MLX template
            # (context first + a meta-instruction about she/it/corrections) made the 1.5B
            # distill spend its entire think parsing the INSTRUCTION instead of computing
            # (composite battery transcript: it re-quoted the instruction 3x, never reached
            # 50-24, answered 27.5). Also relevance-filter the facts so unrelated pins
            # (hotel room number) don't ride into an arithmetic turn.
            rel = mc._sem_matches(user_msg, chunks, self.bge, cap=3, min_sim=0.4) or chunks[-2:]
            rel = mc._with_amendments(rel, chunks)      # corrections ride along...
            rel = mc.mark_superseded(rel)               # ...and are RESOLVED before injection:
            # the 'most recent value wins' instruction does not work on a 1.5B (v3 C7 twice);
            # explicit (outdated)/(current) tags are mechanical to follow.
            multi = bool(re.search(r"\([a-c]\)", user_msg))
            tail = ("Answer EVERY lettered part; end with one line listing each part's result."
                    if multi else "End with the final number.")
            # '(v5: "End with the final number." (singular) made the model stop after ONE
            # sub-part of (a)/(b)/(c) questions — P1 concluded at part (b), P5 at part (a))
            aug = (f"{user_msg}\n\n(Earlier in this conversation: {' ; '.join(rel)})\n"
                   f"Use those earlier values if the question refers to them. Ignore lines "
                   f"marked (outdated). {tail}")
        elif chunks:
            # retrieval confidence picks the template. Measured sims OVERLAP across the
            # boundary (true paraphrase match 0.592 vs blood-type/badge false hit 0.543),
            # so a threshold can't reject false hits without killing paraphrase recall —
            # below 0.65 the template carries an ESCAPE HATCH instead. Without it, the
            # strict 'the answer IS in the Context' premise forced 'your blood type is
            # VB-7731' out of a badge-code chunk (v4 t35).
            qv = self.bge._encode([user_msg], is_query=True)[0]
            sims = self.bge._encode(list(chunks), is_query=False) @ qv
            if float(max(sims)) < 0.62:   # true-match floor measured at 0.653 (hotel)
                # uncertain band: don't ASSERT an answer at all. Nothing separates a true
                # paraphrase match from a false hit here — full-question sims overlap
                # (0.592 vs 0.543), topic sims overlap (0.498 vs 0.487), the in-prompt
                # escape hatch got steamrolled, and the yes/no micro-judge said yes and
                # invented blood type 'A' for a badge code. So show the closest saved note
                # verbatim instead: honest for a false hit, and for a genuine paraphrase
                # the note IS the answer. Mechanical, zero extra latency.
                note = chunks[int(sims.argmax())]
                return (f"I don't have that saved exactly — the closest note I have: "
                        f"\"{note}\"", src, chunks)
            aug = (f"Context (retrieved from {src}): {' ; '.join(chunks)}\n\n"
                   f"Question: {user_msg}\nThe answer is stated EXPLICITLY in the Context above. Do NOT "
                   f"calculate, reason about, or transform it, and ignore anything earlier in the "
                   f"conversation — just read the matching value from the Context and reply with ONLY that "
                   f"value, verbatim (keep letter prefixes/punctuation, e.g. 'EMP-1234' not '1234'; use the "
                   f"most recent value if it was corrected).")
        else:
            aug = user_msg
        quote_recall = bool(chunks) and not compute_like
        check_chunks = [] if compute_like else chunks
        if quote_recall:
            answer = self._clean_quote(aug)
            for _ in range(retries):
                if mc._answer_ok(answer, check_chunks, user_msg):
                    break
                answer = self._clean_quote(aug)
        else:
            snap = (list(self.gen), list(self.kept), self.absorbed)
            answer = self._gen_once(aug)
            for _ in range(retries):
                if mc._answer_ok(answer, check_chunks, user_msg):
                    break
                self.gen, self.kept, self.absorbed = list(snap[0]), list(snap[1]), snap[2]
                answer = self._gen_once(aug)
        if store == "persist":
            self.mem.persist(user_msg)
        elif store == "session" and intent == "fact":
            self.mem.remember_session(user_msg)
        if compute_like and mc._answer_ok(answer, [], user_msg):
            # self-log the RESULT of a compute turn. The follow-up battery showed why: the
            # log carries the previous QUESTION but not its answer, so "I pay with $50,
            # how much change?" forced a full re-derivation of the $24 — and the model
            # tangled the two questions. With "... — result: 24" in the log, the follow-up
            # reads the prior result instead of re-deriving it.
            box = re.findall(r"\\boxed\{([^}]*)\}", answer)
            nums = re.findall(r"\$?\d[\d,]*(?:\.\d+)?", answer)
            val = box[-1].strip() if box else (nums[-1] if nums else None)
            if val:
                # DECLARATIVE form only. Logging "<question> — result: 24" baited the model
                # into re-answering the embedded QUESTION instead of the new one (composite
                # transcript: it ignored the $50-change question entirely and re-derived the
                # muffin total). Interrogative text must never be re-injected as context.
                # And the result line is WORKING STATE, not history: keeping old results
                # alongside made the model hedge between them ('if 24 then 26; if 32 then
                # 18' — composite v2 C3). A recompute supersedes the previous result.
                self.mem.session = [l for l in self.mem.session
                                    if not l.startswith("Earlier computed result:")]
                self.mem.remember_session(f"Earlier computed result: {val}.")
        return answer, src, chunks
