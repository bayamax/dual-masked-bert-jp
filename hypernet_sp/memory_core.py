"""Pure-python memory/grounding core extracted VERBATIM from tiered_rag_mlx.py (which
imports MLX and loads the model at import time, so none of this logic was unit-testable
off-device). Behaviour is unchanged; the BGE encoder is injected instead of lazy-global.

tiered_rag_mlx.py integration: delete the duplicated definitions and
    from memory_core import (_words, _overlap, _best, _matches, _sem_matches, _match,
                             _is_question, _looks_mathy, _is_factlike, wants_persist,
                             is_grounded, _looks_degenerate, _extract_answer, _answer_ok,
                             TieredMemory)
(pass bge=_bge() where the signatures ask for it)."""
import json, os, re

_STOP = set("the a an of in on at to is are was were am my your you i me we what who which when where "
            "why how do does did can could will would should it its this that these those and or but for "
            "with about please tell remember our us so let's let now and have has had get got make made "
            "many much more most some any here there also like want need give given take just it's".split())


_CJK = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]+")     # kana + unified CJK
_JP_STOP = set("です ます した して いる ある する それ これ あれ どこ なに 何で ですか ました".split())


def _words(s):
    """Content tokens for lexical overlap. ASCII words as before; CJK runs additionally
    contribute character BIGRAMS (Japanese has no spaces — without this the lexical tier is
    stone dead on Japanese input: '部屋番号' and '部屋の番号' share 部屋/番号 bigrams)."""
    out = {w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", s.lower()) if w not in _STOP and len(w) >= 2}
    for run in _CJK.findall(s):
        if len(run) == 1:
            continue
        out.update(b for b in (run[i:i + 2] for i in range(len(run) - 1)) if b not in _JP_STOP)
    return out


def _overlap(qw, tw):
    n = 0
    for a in qw:
        if any(a == b or (len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)))
               for b in tw):
            n += 1
    return n


def _best(query, store):
    qw = _words(query)
    scored = sorted(((_overlap(qw, _words(t)), t) for t in store), key=lambda x: -x[0])
    return scored[0] if scored and scored[0][0] > 0 else (0, None)


def _sem_matches(query, store, bge, cap=4, min_sim=0.46, top_delta=0.12, rec_eps=0.05):
    if bge is None or not store:
        return None
    qv = bge._encode([query], is_query=True)[0]
    dv = bge._encode(list(store), is_query=False)
    sims = dv @ qv
    keep = [i for i in range(len(store)) if sims[i] >= min_sim]
    if not keep:
        return []
    hi = max(sims[i] for i in keep)
    keep = [i for i in keep if sims[i] >= hi - top_delta]
    keep.sort(key=lambda i: (-round(float(sims[i]) / rec_eps), -i))
    return [store[i] for i in sorted(keep[:cap])]


def _matches(query, store, cap=4):
    qw = _words(query)
    scored = [(_overlap(qw, _words(t)), i, t) for i, t in enumerate(store)]
    scored = [s for s in scored if s[0] > 0]
    if not scored:
        return []
    hi = max(s[0] for s in scored)
    near = [s for s in scored if s[0] >= hi - 1]
    top = sorted(near, key=lambda s: (-s[0], -s[1]))[:cap]
    return [t for _, _, t in sorted(top, key=lambda s: s[1])]


def _match(query, store, bge=None, cap=4):
    """Semantic (BGE) first for English; for CJK queries the order is REVERSED. Measured on
    bge-small-EN-v1.5: Japanese pos/neg margins collapse to +0.07..0.14 (vs +0.34 English)
    while absolute sims sit ~0.65-0.78 — far above min_sim=0.46 and inside top_delta of the
    best — so the EN-calibrated semantic tier INJECTS DISTRACTORS on Japanese. Lexical CJK
    bigrams are precise; semantic only backstops them, at a raised bar."""
    if _CJK.search(query):
        lex = _matches(query, store, cap)
        if lex:
            return lex
        r = _sem_matches(query, store, bge, cap, min_sim=0.72, top_delta=0.06)
        return r or []
    r = _sem_matches(query, store, bge, cap)
    return _matches(query, store, cap) if r is None else r


_Q_LEAD = ("what", "what's", "whats", "who", "who's", "whose", "when", "where", "which",
           "how", "why", "recall", "remind", "find", "do you know", "tell me", "is my", "are my")


def _is_question(text):
    t = text.strip().lower()
    return t.endswith("?") or t.startswith(_Q_LEAD)


_MATH = re.compile(r"how (much|many)|calculat|comput|percent|%|\bplus\b|\bminus\b|times|divided|"
                   r"average|\btotal\b|\bsum\b|sqrt|\d\s*[-+*/=]\s*\d")


def _looks_mathy(q):
    return bool(re.search(r"\d", q) and _MATH.search(q.lower()))


def wants_persist(text):
    return bool(re.search(r"remember|save|profile|note|for later|always", text.lower()))


_FACTLIKE = re.compile(r"\d|\b(is|are|am|was|were|will|'ll|named|called|theme|colou?r|name|make it|"
                       r"change|instead|order|paint|costs?|each)\b", re.I)


def _is_factlike(text):
    return (not _is_question(text)) and bool(_FACTLIKE.search(text))


def is_grounded(answer, chunks, question=""):
    ctx = " ".join(chunks).lower()
    qw = {w for w in re.findall(r"[a-z0-9\-]+", question.lower()) if len(w) >= 2}
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", answer)
    salient = [t for t in toks
               if (any(c.isdigit() for c in t) or t[0].isupper() or len(t) >= 5)
               and t.lower() not in _STOP and t.lower() not in qw]
    if not salient:
        return False
    return any(t.lower() in ctx for t in salient)


def _looks_degenerate(text):
    w = text.split()
    if len(w) < 8:
        return False
    tri = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    from collections import Counter
    top = Counter(tri).most_common(1)
    return bool(top) and top[0][1] >= 4


def _extract_answer(body):
    if "</think>" not in body:
        return "(still thinking — timed out)"
    after = body.split("</think>")[-1].strip()
    if after:
        return after
    inside = body.split("</think>")[0].replace("<think>", " ")
    boxed = re.findall(r"\\boxed\{([^}]*)\}", inside)
    if boxed and boxed[-1].strip():
        return f"\\boxed{{{boxed[-1].strip()}}}"
    lines = [l.strip() for l in inside.splitlines() if l.strip()]
    return lines[-1] if lines else "(no answer)"


_EMPTY = ("", "(still thinking — timed out)", "(no answer)")


def _answer_ok(answer, chunks, question):
    a = (answer or "").strip()
    if a in _EMPTY or len(a) < 2:
        return False
    if _looks_degenerate(a):
        return False
    if chunks and not is_grounded(answer, chunks, question):
        return False
    return True


class TieredMemory:
    """L1 same-session / L2 prior-session(disk) / pins working-memory. Verbatim from
    tiered_rag_mlx.py with the BGE encoder injected (bge=None -> lexical only)."""
    TH = 1
    PINCAP = 12
    LOGCAP = 16

    def __init__(self, persistent_path, bge=None):
        self.path = persistent_path
        self.bge = bge
        self.session, self.persistent, self.pins = [], [], []
        if os.path.exists(persistent_path):
            with open(persistent_path) as f:
                self.persistent = [json.loads(l)["text"] for l in f if l.strip()]

    def remember_session(self, text):
        self.session.append(text)

    def pin(self, text):
        if text in self.pins:
            self.pins.remove(text)
        self.pins.append(text)
        if len(self.pins) > self.PINCAP:
            self.pins = self.pins[-self.PINCAP:]

    def persist(self, text):
        self.session.append(text)
        self.persistent.append(text)
        with open(self.path, "a") as f:
            f.write(json.dumps({"text": text}) + "\n")

    def retrieve_personal(self, query):
        m1 = _match(query, self.session, self.bge)
        m2 = _match(query, self.persistent, self.bge)
        if m1 or m2:
            chunks = m1 + [c for c in m2 if c not in m1]
            src = "+".join(p for p, ok in [("L1·same-session", m1), ("L2·prior-session", m2)] if ok)
            return src, chunks
        mp = _sem_matches(query, self.pins, self.bge, min_sim=0.5) if self.pins else None
        if mp:
            return "WM·pins", mp
        return None, []
