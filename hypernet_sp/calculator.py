"""Post-hoc arithmetic verification & repair ("calculator tool" for a model that cannot
tool-call).

The R1-distill-1.5B writes correct EXPRESSIONS and then mis-evaluates them (v5 P1:
2000x1.05^3 -> 121550.625; composite B: 650-200 -> 210). It was never trained to emit tool
calls, and every instruction-based intervention failed in testing — so the calculator runs
AFTER generation, mechanically:

  1. find arithmetic claims  `<expr> = <value>`  in the turn text (think + answer);
  2. re-evaluate each with a restricted-AST evaluator (no eval(), numbers/+-*/^()only);
  3. if a claim is wrong AND the answer's final value equals the wrong claim, substitute
     the recomputed value.

Scope (honest): fixes EVALUATION errors. It cannot fix a wrong SETUP (v5 P7 added the
drain instead of subtracting — the expression itself was wrong) and does not touch
non-arithmetic answers.
"""
import ast
import re

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod)


def safe_eval(expr):
    """Evaluate a pure-arithmetic expression. Raises ValueError on anything else."""
    expr = _normalise(expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"unparseable: {expr!r}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"disallowed node {type(node).__name__} in {expr!r}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"non-numeric constant in {expr!r}")
    return eval(compile(tree, "<calc>", "eval"))      # safe: AST whitelisted above


def _normalise(s):
    s = s.replace(",", "").replace("$", "").replace("%", "")
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-")
    s = s.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    s = re.sub(r"\^", "**", s)
    s = re.sub(r"(\d)\s*\(", r"\1*(", s)              # implicit mult: 2(3+4)
    return s.strip()


# an arithmetic claim: expression with >=1 operator and >=2 numbers, then '=', then value.
# Greedy enough for '2000 * 1.05^3 = 2315.25', '650 - 200 = 210', '(1/6+1/4)-1/12 = 12/7'.
_NUM = r"\$?\d[\d,]*(?:\.\d+)?"
_CLAIM = re.compile(
    rf"((?:{_NUM}|[()+\-*/×÷·^\s]|\\times|\\cdot|\\div){{3,}})"
    rf"=\s*({_NUM}(?:\s*/\s*{_NUM})?)")


def _to_value(s):
    s = _normalise(s)
    if "/" in s:                                       # claimed fractions: a/b
        try:
            return safe_eval(s)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def find_claims(text):
    """Yield (expr_str, claimed_value, actual_value, ok) for every verifiable claim."""
    out = []
    for m in _CLAIM.finditer(text):
        expr, claimed_s = m.group(1), m.group(2)
        if not re.search(r"[+\-*/^×÷]|\\times|\\div", expr):
            continue
        if len(re.findall(_NUM, expr)) < 2:
            continue
        claimed = _to_value(claimed_s)
        if claimed is None:
            continue
        try:
            actual = safe_eval(expr)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        # respect the precision the model used: '38' for 38.46 counts as right if the
        # claim is the rounded form
        dec = len(claimed_s.split(".")[1]) if "." in claimed_s else 0
        ok = abs(round(actual, dec) - claimed) <= max(10 ** -dec, abs(actual) * 1e-9) \
            if dec else abs(actual - claimed) < max(0.5, abs(actual) * 1e-9) + 1e-9
        out.append((expr.strip(), claimed, actual, ok))
    return out


def _fmt(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{round(v, 6):g}"


def repair_answer(answer, full_body=None):
    """If the answer's value(s) equal the RHS of a WRONG arithmetic claim (searched in the
    full turn body when given, else the answer itself), substitute the recomputed value.
    Returns (fixed_answer, corrections) — corrections = [(expr, claimed, actual)]."""
    scope = full_body if full_body is not None else answer
    wrong = [(e, c, a) for e, c, a in
             ((e, c, a) for e, c, a, ok in find_claims(scope) if not ok)]
    if not wrong:
        return answer, []
    fixed, corrections = answer, []
    for expr, claimed, actual in wrong:
        # match the claimed value verbatim: repr keeps full precision ('121550.625');
        # '%g' would round to 6 significant digits and miss it
        cands = {repr(claimed).rstrip("0").rstrip("."), f"{claimed:g}", _fmt(claimed)}
        norm = fixed.replace(",", "")
        for cstr in cands:
            if cstr and cstr in norm:
                fixed = re.sub(re.escape(cstr), _fmt(actual), norm)
                corrections.append((expr, claimed, actual))
                break
    return fixed, corrections
