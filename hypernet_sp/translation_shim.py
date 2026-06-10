"""Translation shim with verbatim-span protection (the JP->EN-only-inside architecture).

Flow: user JP -> [inbound] -> EN pipeline (intent / retrieval / CoT / guards, all EN)
      -> EN final answer -> [outbound] -> JP display. The CoT is never translated.

The one thing a translator must NOT touch is the system's core invariant: verbatim values.
"合言葉は「さくら」" -> "The passphrase is 'Sakura'" -> stored EN -> recalled -> re-rendered
さくら/サクラ/桜 — the surface the user typed is gone. Codes survive by luck, JP-native
strings don't. So: extract protected spans BEFORE translation, swap in placeholders the NMT
copies through, restore the ORIGINAL surface after. Memory then stores EN sentences with
true-original spans embedded, and quoting returns exactly what the user typed.

The translator is injected (`translate_fn(text, src, tgt) -> text`): Apple Translation
framework on-device, opus-mt elsewhere, a mock in tests. If a placeholder does not survive
the round trip (NMT dropped it), the missing values are appended rather than lost.
"""
import re

# what must survive verbatim: quoted strings (JP and latin quotes), codes/ids, money/
# numbers with units, latin runs embedded in JP text, times. The specificity probe can be
# layered on top via `extra_spans`; this regex floor needs no model.
_PROTECT = re.compile(
    # NB: \b is useless next to kana/kanji (they are \w in unicode re) — "はPOL-55821です"
    # has no word boundary at 'P'. Latin/digit lookarounds delimit instead.
    r"「[^」]{1,40}」|『[^』]{1,40}』|\"[^\"]{1,40}\"|'[^']{1,40}'"      # quoted strings
    r"|(?<![A-Za-z0-9])[A-Za-z]+(?:-[A-Za-z0-9]+)+(?![A-Za-z0-9])"      # hyphen codes (POL-55821)
    r"|\$?\d[\d,]*(?:\.\d+)?\s?(?:%|円|cm|mm|km|kg|am|pm)?"             # numbers/money/units
    r"|(?<![A-Za-z0-9])[A-Za-z]*\d[A-Za-z0-9\-]*(?![A-Za-z0-9])"        # codes (QX7-2291)
    r"|(?<![A-Za-z0-9])[A-Z][A-Za-z0-9\-]{2,}(?![A-Za-z0-9])")          # latin proper runs

# placeholder the NMT treats as an opaque token and copies through. Digits-in-brackets
# survive mainstream NMT far more reliably than exotic unicode.
_PH = "[[{}]]"
_PH_FIND = re.compile(r"\[\[\s?(\d+)\s?\]\]")


def protect(text, extra_spans=()):
    """Return (masked_text, mapping). extra_spans: probe-detected spans to protect too."""
    spans = list(dict.fromkeys([m.group(0) for m in _PROTECT.finditer(text)] + list(extra_spans)))
    spans.sort(key=len, reverse=True)                # longest first so substrings don't clobber
    mapping, out = {}, text
    for i, s in enumerate(spans):
        if s not in out:
            continue
        ph = _PH.format(i)
        mapping[i] = s
        out = out.replace(s, ph)
    return out, mapping


def restore(text, mapping):
    """Swap placeholders back to original surfaces. Returns (text, ok). Placeholders the
    translator dropped are appended as a parenthetical so the VALUE is never lost."""
    seen = set()

    def sub(m):
        i = int(m.group(1))
        seen.add(i)
        return mapping.get(i, m.group(0))
    out = _PH_FIND.sub(sub, text)
    missing = [mapping[i] for i in mapping if i not in seen]
    if missing:
        out = out.rstrip() + " (" + ", ".join(missing) + ")"
    return out, not missing


class TranslationShim:
    def __init__(self, translate_fn, user_lang="ja", core_lang="en"):
        self.tr, self.user_lang, self.core_lang = translate_fn, user_lang, core_lang

    def inbound(self, user_text, extra_spans=()):
        """User language -> core EN, verbatim spans preserved byte-for-byte."""
        masked, mapping = protect(user_text, extra_spans)
        en = self.tr(masked, self.user_lang, self.core_lang)
        out, ok = restore(en, mapping)
        return out

    def outbound(self, en_answer, extra_spans=()):
        """Core EN final answer -> user language for display. Run AFTER the groundedness
        gate (guards must see the EN text the model actually produced)."""
        masked, mapping = protect(en_answer, extra_spans)
        jp = self.tr(masked, self.core_lang, self.user_lang)
        out, ok = restore(jp, mapping)
        return out
