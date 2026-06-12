"""Conditioned-confidence intent routing (STATUS 2026-06-10, next-step 3).

Problem: `tiered_rag_mlx.intent_of` accepts the BGE+LogisticRegression label only when
top-1 probability >= 0.5; anything below falls ALL the way back to the crude regex gates,
whose "question + I/my -> recall" rule mis-routes e.g. "any tips for my weekend trip to
Kyoto?" (chitchat/lookup) into personal-memory recall.

Fix (this module): a THREE-band decision instead of a binary one.
  p1 >= hi              -> trust the classifier outright (unchanged behaviour).
  lo <= p1 < hi         -> GREY ZONE: the classifier is unsure between its top-2 classes,
                           but those two are almost always the right candidates (top-2
                           coverage is ~99% on held-out data). Use the cheap syntactic
                           signals only to ARBITRATE BETWEEN the top-2 — never to jump to
                           an arbitrary third class the way the full regex fallback can.
  p1 < lo / no model    -> full regex fallback (unchanged: it's the no-information case).

The module is import-safe without MLX (pure stdlib + the caller-supplied clf/bge), so
`tiered_rag_mlx.py` and the evals can share one implementation.

Integration (tiered_rag_mlx.py):
    from intent_route import route_intent
    def intent_of(text, min_conf=0.5):
        return route_intent(text, _intent_clf(), _bge(), hi=min_conf)
"""
import re

# ---- regex gates: kept byte-identical to tiered_rag_mlx.py so behaviour can't drift ----
_Q_LEAD = ("what", "what's", "whats", "who", "who's", "whose", "when", "where", "which",
           "how", "why", "recall", "remind", "find", "do you know", "tell me", "is my", "are my")

_MATH = re.compile(r"how (much|many)|calculat|comput|percent|%|\bplus\b|\bminus\b|times|divided|"
                   r"average|\btotal\b|\bsum\b|sqrt|\d\s*[-+*/=]\s*\d")

_FACTLIKE = re.compile(r"\d|\b(is|are|am|was|were|will|'ll|named|called|theme|colou?r|name|make it|"
                       r"change|instead|order|paint|costs?|each)\b", re.I)

_COMMAND = re.compile(r"\b(recompute|recalculate|recalc|redo|new total)\b|the total", re.I)


def is_question(text):
    t = text.strip().lower()
    return t.endswith("?") or t.startswith(_Q_LEAD)


def looks_mathy(q):
    return bool(re.search(r"\d", q) and _MATH.search(q.lower()))


def _is_factlike(text):
    return (not is_question(text)) and bool(_FACTLIKE.search(text))


def regex_intent(text):
    """The original full fallback (verbatim port of tiered_rag_mlx.intent_of's tail)."""
    if looks_mathy(text):
        return "math"
    if _COMMAND.search(text.lower()):
        return "command"
    if is_question(text):
        return "recall" if re.search(r"\b(my|mine|i|me|we|our)\b", text, re.I) else "lookup"
    return "fact" if _is_factlike(text) else "chitchat"


# ---- grey-zone arbitration ------------------------------------------------------------
# Per-class syntactic evidence. These only VOTE between the classifier's top-2 candidates;
# a class with no rule simply never receives a vote (the classifier order then decides).
#
# recall must reference STORED user info, so a bare "I/my" is NOT enough (that incidental
# pronoun is exactly the STATUS failure: "tips for my weekend trip?" -> recall). It needs a
# memory verb ("did I say", "remind me") or the value-of-my-attribute shape ("what's my X",
# "when is my flight") — a WH-word with "my/I" inside the same clause.
_RECALL_CUE = re.compile(r"\b(remind me|recall|do you remember|did (i|you) (say|mention|tell|leave|park)|"
                         r"what (was|did) (i|you)\b|again\b)|"
                         # 'how' joins the WH+my branch only in its quantity forms:
                         # "how old is my daughter" = recall, but "How can I make my phone
                         # battery last longer?" is a how-to (v7 H1 was hijacked to recall)
                         r"\b(what|whats|what's|which|when|where|how (?:many|much|old|long))\b"
                         r"[^,.!?]{0,24}\b(my|mine)\b", re.I)
# advice/ideas-seeking questions are open-ended generation (chitchat), not retrieval
_ADVICE = re.compile(r"\b(tips?|suggest(ions?)?|recommend(ations?)?|ideas?|should (i|we)|"
                     r"what do you think|brainstorm|help me (plan|write|pick|decide)|good)\b", re.I)
# imperative TEXT-ARTIFACT requests are generation, never fact/command ("Write a short
# polite email..." was fact-acked 'Got it — saved.' in v7; the summarise-this-passage and
# make-it-shorter turns likewise). "make it <digit>..." stays a correction-fact.
_CREATIVE = re.compile(r"^(?:please\s+)?(?:write|draft|compose|create|rewrite|rephrase|"
                       r"reword|summari[sz]e|shorten|expand|translate|edit|polish|improve)\b"
                       r"|\bmake (?:it|this|that)\b(?![^.!?]*\d)", re.I)
# first-person emotional shares need a RESPONSE, not a silent ack (v7 E1: venting got
# 'Got it — saved.')
_EMOTIONAL = re.compile(r"\b(i(?:'m| am| was| had|'ve had)?\b[^.!?]{0,40}\b"
                        r"(feel|feeling|felt|rough|tough|stressed|exhausted|tired|sad|"
                        r"happy|excited|frustrated|overwhelmed|anxious|vent|venting))", re.I)
_VERIFY = re.compile(r"\b(verify|double-?check|fact-?check|confirm)\b", re.I)
_NUMWORD = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                      r"twenty|thirty|forty|fifty|hundred|thousand|dozen|half|quarter|double)\b", re.I)
_OPWORD = re.compile(r"\b(plus|minus|times|divided|multiply|multiplied|subtract(ed)?|add(ed|s)?|percent|"
                     r"sum of|total of|product of|difference|average of|squared|cubed|double|half of)\b", re.I)


def _evidence(text, label):
    """Weighted syntactic support for `label`: 2 = a strong positive cue, 1 = a weak default
    shape, 0 = nothing. Only used to arbitrate between the classifier's top-2 candidates."""
    t = text.strip().lower()
    q = is_question(text)
    if label == "math":
        return 2 * int(looks_mathy(text) or (bool(_NUMWORD.search(text)) and
                                             bool(_MATH.search(t) or _OPWORD.search(t))))
    if label == "command":
        return 2 * int(bool(_COMMAND.search(t)))
    if label == "recall":
        return 2 * int(q and bool(_RECALL_CUE.search(text)))
    if label == "lookup":
        return 0          # no reliable positive surface cue — reachable as top-1 or via the fact-fix
    if label == "fact":
        return int(_is_factlike(text))
    if label == "chitchat":
        if q and _ADVICE.search(text):
            return 2      # ideas/advice-seeking question -> open-ended generation
        return int(not q and not _is_factlike(text))
    return 0


def route_intent(text, clf_bundle, bge, hi=0.5, lo=0.30):
    """Three-band intent decision. `clf_bundle` is the joblib dict {"clf": LogisticRegression},
    `bge` an object with ._encode([text], is_query=False) -> (1, 384). Either may be None."""
    if not clf_bundle or not bge:
        return regex_intent(text)
    try:
        x = bge._encode([text], is_query=False)[0]
        p = clf_bundle["clf"].predict_proba([x])[0]
        classes = list(clf_bundle["clf"].classes_)
    except Exception:
        return regex_intent(text)
    if _CREATIVE.search(text) and not looks_mathy(text):
        # generation request, whatever the clf says — UNLESS it carries a math cue
        # ("Write out the calculation of 17*24" must keep the compute path: forced think,
        # convergence policy, tags and the calculator all live there)
        return "chitchat"
    if _EMOTIONAL.search(text) and not is_question(text):
        return "chitchat"                          # respond with empathy (caller may still log)
    if _VERIFY.search(text):
        # "verify/confirm/double-check the number" means GO CHECK THE SOURCE -> lookup.
        # Routed to recall it dead-ends in honest-miss (live oil-price test); as lookup,
        # known-fact-first serves stored values and the web path gets the value-anaphora
        # expansion ("... ($86.78)") that defect 17 built.
        return "lookup"
    order = sorted(range(len(p)), key=lambda i: -p[i])
    top1, top2 = str(classes[order[0]]), str(classes[order[1]])
    p1 = float(p[order[0]])

    def _fix_interrogative_fact(lab):
        # an interrogative is never a fact. The shipping rule sent every such question to
        # recall, which is how "how do I get from Kyoto station to Kinkaku-ji?" (clf says
        # fact, p=0.52) ended up probing personal memory. Route by the actual cues instead.
        if lab == "fact" and is_question(text):
            if looks_mathy(text):
                return "math"
            if _RECALL_CUE.search(text):
                return "recall"
            if _ADVICE.search(text):
                return "chitchat"
            return "lookup"
        return lab

    if p1 >= hi:
        return _fix_interrogative_fact(top1)
    if p1 >= lo:
        # grey zone: stay inside the classifier's top-2; syntax arbitrates
        e1, e2 = _evidence(text, top1), _evidence(text, top2)
        if e1 == e2 == 0 and is_question(text) and _ADVICE.search(text) \
                and not _RECALL_CUE.search(text) and not looks_mathy(text):
            return "chitchat"                  # ideas-seeking question neither candidate claims
        lab = top2 if e2 > e1 else top1
        return _fix_interrogative_fact(lab)
    return regex_intent(text)
