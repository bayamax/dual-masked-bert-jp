"""Anaphora resolution for L3 web queries (STATUS 2026-06-10, next-step 2 — "cheapest option").

Problem: a follow-up like "what are the best temples to visit there?" goes to DuckDuckGo
verbatim; without the antecedent ("Kyoto", introduced a turn earlier and living in working
memory) the engine returns whatever city ranks best (Osaka temples).

Fix: if a lookup query contains an unresolved referring expression AND no anchor entity of
its own, append the most recent pinned/logged entities to the web query string:

    "best temples to visit there?"  ->  "best temples to visit there? (Kyoto)"

Resolution never rewrites the pronoun (no parsing, no model); search engines treat the
parenthetical as extra keywords, which is all the disambiguation they need.

Integration (tiered_rag_mlx.py, the `lookup` branch and `_web_retrieve` fallback):
    from anaphora import expand_web_query
    name, ch = _web_retrieve(expand_web_query(user_msg, self.mem.pins, self.mem.session))
"""
import re

# referring expressions that need an antecedent: place/object/person pronouns + bare deictic
# noun phrases ("the place", "the city"). \b keeps "their"/"theory" etc. from matching.
_ANAPHOR = re.compile(r"\b(there|it|that one|that place|this place|them|they|he|she|him|her|its|"
                      r"the (place|city|town|area|spot|venue|hotel|restaurant|museum|station|one|"
                      # value anaphora (trigger report finding B): 'verify the number' went
                      # to the web naked and matched phone-verification spam
                      r"number|total|result|value|price|amount|figure|answer|rate))\b", re.I)

# an entity ALREADY in the query makes expansion unnecessary (and harmful: appending Kyoto
# to "temples in Osaka?" would poison the search) — a proper-noun run past position 0, or a
# code / number anywhere.
_CODEISH = re.compile(r"\b[A-Za-z]*\d[A-Za-z0-9\-]*\b")
# concrete VALUES (money, decimals, big integers) — the antecedent of value anaphora
# ('verify the number' -> append the $82.40 the user was just told about)
_VALUE = re.compile(r"\$\d[\d,]*(?:\.\d+)?|\b\d+\.\d+\b|\b\d{3,}\b")
_PROPER = re.compile(r"\b[A-Z][a-z]+(?:[\s\-][A-Z][a-z]+){0,2}\b")   # 'Lisa Su' ('Su' is 2 chars)

# generic capitalised words that are not anchors (sentence starters, weekdays, months, "I")
_GENERIC = set("the this that what who when where which how why can could should would do does did is are was "
               "i my our we you your please also and but yes no okay monday tuesday wednesday thursday friday "
               "saturday sunday january february march april may june july august september october november "
               "december am pm ok hey hi hello thanks earlier today yesterday tomorrow".split())


def _entities(text):
    """Proper-noun runs + codes in `text`, in order, minus generic capitalised words and
    anything sentence-initial that is generic-shaped. Mirrors the cheap-candidate philosophy
    of tiered_rag_mlx._CAND: regex enumerates, a probe (optional) judges."""
    ents = []
    for m in _PROPER.finditer(text):
        w = m.group(0)
        if w.lower() in _GENERIC or all(p.lower() in _GENERIC for p in w.split()):
            continue
        ents.append(w)
    for m in _VALUE.finditer(text):
        ents.append(m.group(0))
    for m in _CODEISH.finditer(text):
        w = m.group(0)
        if len(w) >= 2 and not w.isdigit() or (w.isdigit() and len(w) >= 3):
            ents.append(w)
    return list(dict.fromkeys(ents))


def has_own_anchor(query):
    """True if the query already names a concrete entity (so expansion must not run)."""
    if _CODEISH.search(query):
        return True
    for m in _PROPER.finditer(query):
        if m.start() == 0:                      # sentence-initial capitalisation is not evidence
            continue
        if m.group(0).lower() in _GENERIC or all(p.lower() in _GENERIC for p in m.group(0).split()):
            continue
        return True
    return False


def needs_resolution(query):
    """An unresolved referring expression, and no anchor entity of the query's own."""
    return bool(_ANAPHOR.search(query)) and not has_own_anchor(query)


def expand_web_query(query, pins, session=(), cap=2, scorer=None):
    """Return `query`, or `query (Entity1, Entity2)` when it needs an antecedent.

    pins/session: most-recent-LAST lists of past turns (TieredMemory.pins / .session).
    cap: max entities appended (keep the keyword payload small — engines AND terms).
    scorer: optional callable(entities, query) -> ranked entities (e.g. BGE relevance);
            default ranking is pure recency (the most recently mentioned entity wins,
            matching the 'corrections win' convention used elsewhere in the runtime)."""
    if not needs_resolution(query):
        return query
    ents = []
    for turn in list(reversed(list(pins))) + list(reversed(list(session))):
        for e in _entities(turn):
            if e.lower() not in query.lower() and e not in ents:
                ents.append(e)
    if not ents:
        return query
    ents = scorer(ents, query)[:cap] if scorer else ents[:cap]
    return f"{query} ({', '.join(ents)})"
