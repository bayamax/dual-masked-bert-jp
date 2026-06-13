"""cotscan_live.py — COTSCAN_V1 残作業 #1+#2 の実検証(生成ループ内 per-position recall)。

COTSCAN_SPEC.md は「CoT 途中で隠れ表現を手がかりに、SP 圧縮で露出窓から外れた
具体事実を引き戻せるか」を *教師強制* で検証した。本スクリプトはその残作業:
  #1 per-position ゲートを生成ループに配線(mid-stream recall)
  #2 実生成 CoT(R1 長考)での再検証
を行う。app_session_torch._gen_once のブロック再構築構造
  block = [SP(pooler over kept)] + [recall(evicted blocks)] + [raw window(last rw)]
を最小再現し、想起クエリの出どころだけを 3 アームで切り替えて比較する:

  OFF  : recall 無し(rec_emb 常に None)。SP 圧縮で具体値は失われるはず(対照)。
  TURN : ターン頭で 1 回だけ、ユーザ質問をクエリに想起(= 現行 BLOCKRECALL/本番)。
  MID  : 各再構築で「いま生成中の CoT 末尾」をクエリに想起し直す(= COTSCAN 本題)。

想起本体は block_recall.BlockArchive(学習不要・モデル QK 空間の pre-RoPE クエリ×
ブロックキー)。pooler は本番と同じ AttnPoolSP。判定 = 最終回答に植えた厳密値が出るか。
"""
import os, sys, json, re, time
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import DynamicCache
except Exception:
    from transformers.cache_utils import DynamicCache

from block_recall import BlockArchive
from attn_export3_torch import load_pooler

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.environ.get("FFT_HF", "fft_hf")
RW, C, CAP, TEMP, RECALL_K = 512, 64, 400, 0.6, 2
LAYERS = (8, 14, 20)
SEED = int(os.environ.get("COTSCAN_SEED", "0"))

LOG = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


# ----------------------------------------------------------------------------------------
# Battery: each scenario plants ONE concrete high-entropy value, buries it in filler so it
# lands well outside the rw=512 window (=evicted / SP-only), then asks for it.
#   direct   = the question names the entity (turn-start recall already has the cue).
#   deferred = the question forces a couple of reasoning steps first; the value is only
#              *needed* mid-CoT, which is where MID (per-position) is supposed to win.
# ----------------------------------------------------------------------------------------
FILLER = (
    "The quarterly logistics review covered routine warehouse throughput, staff rotation "
    "schedules, the cafeteria menu refresh, parking lot repaving, the new visitor badge "
    "policy, fire-drill timing, recycling pickup days, the lobby plant rotation, printer "
    "toner restocking, and the annual team photo. None of these items changed any account "
    "numbers, reference codes, or measured values; they were administrative notes only. "
)
def pad(n):  # ~n repeats of filler to bury the fact
    return (FILLER * n)

# gold      = the string whose presence in the final answer = task success.
# operand   = the planted, SP-evicted value that recall must bring back (== gold for
#             direct; the un-named factor for deferred). operand recovery anywhere in the
#             turn output isolates "did recall work" from "did the model then compute".
BATTERY = [
    {
        "id": "ref_code_direct", "kind": "direct",
        "doc": "Note: the internal reference code for the Zephyr account is 7731-Q. " + pad(9),
        "q": "What is the internal reference code for the Zephyr account? "
             "Answer with the exact code.",
        "operand": "7731-Q", "gold": "7731-Q",
    },
    {
        "id": "price_direct", "kind": "direct",
        "doc": "Record: the calibrated unit price of part KX-204 is 3829 yen. " + pad(9),
        "q": "What is the calibrated unit price of part KX-204? Give the exact number.",
        "operand": "3829", "gold": "3829",
    },
    {
        "id": "dimension_deferred", "kind": "deferred",
        "doc": "Spec: the beam segment length is 412 cm. " + pad(9),
        "q": "A bracket spans exactly three beam segments end to end. Reason step by step "
             "and give the total length in cm.",
        "operand": "412", "gold": "1236",   # 412 * 3 ; operand only needed mid-CoT
    },
    {
        "id": "count_deferred", "kind": "deferred",
        "doc": "Inventory: each sealed crate holds 168 units. " + pad(9),
        "q": "We shipped five sealed crates. Work it out step by step: how many units "
             "in total?",
        "operand": "168", "gold": "840",    # 168 * 5
    },
]


def build_model():
    log(f"[load] device={DEV} model={MODEL_DIR}")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    try:
        llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
    except TypeError:
        llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    llm = llm.eval().to(DEV)
    pooler = load_pooler().to(DEV).eval()
    embT = llm.get_input_embeddings()
    torch.manual_seed(SEED)
    return tok, llm, pooler, embT


def emb(embT, ids):
    if ids:
        return embT(torch.tensor([ids], device=DEV))
    H = embT.weight.shape[1]
    return torch.zeros(1, 0, H, dtype=embT.weight.dtype, device=DEV)


def gold_block_index(archive, tok, gold):
    """Which sealed block (decoded) contains the planted value -> retrieval ground truth."""
    for i, b in enumerate(archive.blocks):
        if gold in tok.decode(b["ids"]):
            return i
    return -1


def _topk_idx(scores, k):
    return set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])


def _record_fire(fires, glen, archive, gi, k):
    sc = getattr(archive, "_last_scores", []) or []
    top = max(range(len(sc)), key=lambda i: sc[i]) if sc else -1
    if 0 <= gi < len(sc):
        rank = 1 + sum(1 for s in sc if s > sc[gi])          # 1 = best
        in_topk = gi in _topk_idx(sc, k)
        gscore = round(sc[gi], 3)
    else:
        rank, in_topk, gscore = None, False, None             # gold not a sealed block
    fires.append((glen, top, round(sc[top], 3) if sc else None, gscore, rank, in_topk))


@torch.no_grad()
def run_turn(tok, llm, pooler, embT, doc, aug, gold, operand, arm):
    """One real CoT turn under `arm`. Returns dict with answer + recall diagnostics.

    Faithful to app_session_torch._gen_once: every C tokens we crop the cache back to the
    BOS prime and re-encode block = [SP]+[recall]+[window]. The ONLY thing the arm changes
    is what query drives `recall` and when it refreshes.
    """
    THINK_OPEN = tok.encode("<think>\n", add_special_tokens=False)
    eos = tok.eos_token_id

    # prior context = the document (already "said"); absorb all but the last rw tokens so
    # the planted fact is evicted into the archive and only SP-summarised in the window.
    gen = tok.encode(doc, add_special_tokens=False)
    archive = BlockArchive(llm, tok, layers=LAYERS, block=128, mode="qk")
    R0 = min(len(gen), RW)
    nd_end = len(gen) - R0
    archive.extend(gen[:nd_end])
    kept = list(gen[:nd_end])
    absorbed = nd_end
    gi = gold_block_index(archive, tok, gold)

    feed = tok.encode(f"<｜User｜>{aug}<｜Assistant｜>", add_special_tokens=False) + list(THINK_OPEN)
    turn_start = len(gen)

    cache = DynamicCache()
    bos = tok.bos_token_id if tok.bos_token_id is not None else eos
    prime = llm(input_ids=torch.tensor([[bos]], device=DEV),
                past_key_values=cache, use_cache=True)
    prime_last = prime.logits[:, -1, :].float()
    MQ = cache.get_seq_length()

    rec_emb = None
    rec_ids = []
    # per rebuild: (gen_len, top_block, top_score, gold_score, gold_rank, gold_in_topk)
    fires = []
    fi = new = 0
    in_think = True
    done = False

    def recall_query_ids(mode):
        if mode == "TURN":
            return tok.encode(aug, add_special_tokens=False)
        # MID: the CoT generated so far this turn (tail), which is what the spec scans.
        tail = gen[turn_start:]
        if len(tail) < 4:                      # nothing thought yet -> seed with question
            return tok.encode(aug, add_special_tokens=False)
        return tail[-48:]

    while not done:
        c0 = len(gen); R = min(c0, RW); nd_end = c0 - R
        if nd_end > absorbed:
            archive.extend(gen[absorbed:nd_end])
            kept.extend(gen[absorbed:nd_end]); absorbed = nd_end

        # --- recall, arm-dependent --------------------------------------------------
        if arm == "OFF":
            rec_emb = None
        elif arm == "TURN":
            if rec_emb is None and (archive.blocks or len(archive.buf) >= 16):
                rid = archive.retrieve(recall_query_ids("TURN"), k=RECALL_K)
                if rid:
                    rec_ids = rid; rec_emb = emb(embT, rid)
                    _record_fire(fires, len(gen), archive, gi, RECALL_K)
        elif arm == "MID":
            # per-position: refresh recall EVERY rebuild from the current CoT tail.
            if archive.blocks or len(archive.buf) >= 16:
                rid = archive.retrieve(recall_query_ids("MID"), k=RECALL_K)
                if rid:
                    rec_ids = rid; rec_emb = emb(embT, rid)
                _record_fire(fires, len(gen), archive, gi, RECALL_K)

        # --- rebuild block and step C tokens ---------------------------------------
        parts = []
        if kept:
            parts.append(pooler(emb(embT, kept).float()).to(embT.weight.dtype))
        if rec_emb is not None:
            parts.append(rec_emb.to(embT.weight.dtype))
        if R > 0:
            parts.append(emb(embT, gen[c0 - R:c0]))
        cache.crop(MQ)
        if parts:
            block = torch.cat(parts, 1)
            last = llm(inputs_embeds=block, past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        else:
            last = prime_last

        for _ in range(C):
            if fi < len(feed):
                t = feed[fi]; fi += 1
            else:
                T = TEMP
                t = int(torch.multinomial(F.softmax(last[0] / T, -1), 1))
                if t == eos:
                    done = True; break
                new += 1
                if new >= CAP:
                    done = True; break
            gen.append(t)
            if in_think and "</think>" in tok.decode(gen[-8:]):
                in_think = False
            last = llm(inputs_embeds=emb(embT, [t]), past_key_values=cache,
                       use_cache=True).logits[:, -1, :].float()
        if done:
            break

    body = tok.decode(gen[turn_start:]).split("<｜Assistant｜>", 1)[-1]
    ans = body.split("</think>")[-1].strip() if "</think>" in body else body.strip()
    return {
        "arm": arm,
        "operand_recovered": operand in body,   # did recall bring the evicted value back?
        "gold_in_answer": gold in ans,          # task success (final answer)
        "gold_anywhere": gold in body,
        "retrieval_ok": any(f[5] for f in fires),   # gold block within injected top-k
        "gold_best_rank": min([f[4] for f in fires if f[4] is not None], default=None),
        "operand": operand, "gold": gold,
        "gold_block": gi, "n_blocks": len(archive.blocks),
        "fires": fires, "rec_ids_decoded": tok.decode(rec_ids)[:120] if rec_ids else "",
        "answer": ans[:400], "body_tokens": len(gen) - turn_start,
    }


def main():
    t0 = time.time()
    out_dir = os.environ.get("OUT_DIR", "results_cotscan_live")
    os.makedirs(out_dir, exist_ok=True)
    tok, llm, pooler, embT = build_model()
    results = []
    arms = ["OFF", "TURN", "MID"]
    for sc in BATTERY:
        log(f"\n=== scenario {sc['id']} ({sc['kind']}) gold={sc['gold']} ===")
        per = {"id": sc["id"], "kind": sc["kind"], "gold": sc["gold"], "arms": {}}
        for arm in arms:
            r = run_turn(tok, llm, pooler, embT, sc["doc"], sc["q"], sc["gold"],
                         sc["operand"], arm)
            per["arms"][arm] = r
            log(f"  [{arm:4}] operand_recovered={r['operand_recovered']} "
                f"gold_in_answer={r['gold_in_answer']} retrieval_ok={r['retrieval_ok']} "
                f"gold_best_rank={r['gold_best_rank']} gold_block={r['gold_block']}/"
                f"{r['n_blocks']} fires={len(r['fires'])}")
            log(f"         ans: {r['answer'][:160]!r}")
            # checkpoint after every arm so the HF-side log shows live progress
            results_snapshot(out_dir, results + [per], t0)
        results.append(per)
    summary = summarize(results)
    log("\n=== SUMMARY ===")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    results_snapshot(out_dir, results, t0, summary=summary, final=True)
    log(f"\nDONE t={time.time()-t0:.0f}s")


def summarize(results):
    arms = ["OFF", "TURN", "MID"]
    agg = {a: {"operand": 0, "gold": 0, "ret": 0} for a in arms}
    n = len(results)
    for r in results:
        for a in arms:
            ar = r["arms"].get(a)
            if not ar:
                continue
            agg[a]["operand"] += int(ar["operand_recovered"])
            agg[a]["gold"] += int(ar["gold_in_answer"])
            agg[a]["ret"] += int(ar["retrieval_ok"])
    return {a: f"operand_recovered {agg[a]['operand']}/{n} | task_correct {agg[a]['gold']}/{n}"
               f" | gold_block_in_topk {agg[a]['ret']}/{n}" for a in arms}


def results_snapshot(out_dir, results, t0, summary=None, final=False):
    payload = {"elapsed_s": round(time.time() - t0, 1), "device": DEV, "seed": SEED,
               "config": {"rw": RW, "C": C, "cap": CAP, "temp": TEMP, "recall_k": RECALL_K,
                          "layers": LAYERS}, "results": results}
    if summary:
        payload["summary"] = summary
    if final:
        payload["final"] = True
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "live.log"), "w") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()
