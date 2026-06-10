"""Unit tests for translation_shim.py — pure stdlib, translator mocked two ways:
a well-behaved NMT (copies placeholders) and a hostile one (mangles everything it sees,
proving the protection is what preserves the values).  python3 test_translation_shim.py"""
import re
from translation_shim import TranslationShim, protect, restore


def good_nmt(text, src, tgt):
    """Copies [[i]] placeholders through; 'translates' the rest by tagging it."""
    return f"<{src}->{tgt}> {text}"


def hostile_nmt(text, src, tgt):
    """Worst-case translator: transliterates kana, reformats numbers, rewrites quotes —
    everything OUTSIDE placeholders gets destroyed."""
    t = re.sub(r"[ぁ-んァ-ン一-龯]", "X", text)              # kana/kanji -> mush
    t = re.sub(r"(?<!\[)\b(\d[\d,]*\.?\d*)\b(?!\])", lambda m: m.group(1).replace(",", ""), t)
    return f"<{src}->{tgt}> {t}"


def lossy_nmt(text, src, tgt):
    """Drops the SECOND placeholder entirely (NMT sometimes eats tokens)."""
    return re.sub(r"\[\[1\]\]", "", good_nmt(text, src, tgt))


def check(name, cond, info=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name}{('  | ' + info) if info and not cond else ''}")
    return bool(cond)


def main():
    r = []
    print("== span extraction ==")
    m, mp = protect("合言葉は「さくら」で、保険番号はPOL-55821です。$1,157.63を送金。")
    r.append(check("jp quote protected", "「さくら」" in mp.values()))
    r.append(check("code protected", "POL-55821" in mp.values()))
    r.append(check("money protected", any("1,157.63" in v for v in mp.values())))
    r.append(check("no raw values left in masked text",
                   "さくら" not in m and "55821" not in m and "1157" not in m.replace(",", "")))

    print("== round-trip survival under a HOSTILE translator ==")
    shim = TranslationShim(hostile_nmt)
    en = shim.inbound("合言葉は「さくら」、部屋番号は1408、コードはQX7-2291です")
    r.append(check("jp-native string survives inbound", "「さくら」" in en, en))
    r.append(check("room number survives", "1408" in en))
    r.append(check("code survives", "QX7-2291" in en))
    jp = shim.outbound('Your passphrase is 「さくら」 and the total is $1,157.63.')
    r.append(check("outbound keeps original surfaces", "「さくら」" in jp and "$1,157.63" in jp, jp))

    print("== dropped-placeholder fallback (value must never be lost) ==")
    masked, mp = protect("code QX7-2291 and pass 「さくら」")
    translated = lossy_nmt(masked, "en", "ja")
    out, ok = restore(translated, mp)
    r.append(check("missing value appended, not lost",
                   "さくら" in out or "QX7-2291" in out))
    r.append(check("loss is signalled", ok is False or all(v in out for v in mp.values())))

    print("== plain text passes through ==")
    shim = TranslationShim(good_nmt)
    r.append(check("no spans -> just translated",
                   shim.inbound("こんにちは、調子はどう?").startswith("<ja->en>")))

    print(f"\n{sum(r)}/{len(r)} passed")
    assert all(r), "translation shim tests failed"
    print("TEST_TRANSLATION_SHIM_DONE")


if __name__ == "__main__":
    main()
