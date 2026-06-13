# COTSCAN live — mid-CoT hidden-representation recall (残作業 #1 + #2)

`hypernet-sp-distill`(HF)の `hypernet_sp/COTSCAN_SPEC.md` が **残作業** として挙げた
2 点を実装・実検証したもの:

- **#1 per-position ゲートの配線**: ターン頭 1 回だった想起を、生成ループ内で
  各ブロック再構築ごとに「いま生成中の CoT 末尾」を手がかりに引き直す(mid-stream recall)。
- **#2 実生成 CoT での検証**: 元の検証は教師強制。ここでは R1(DeepSeek-R1-Distill-Qwen-1.5B,
  fft 版)の実生成 CoT で検証する。

## 何を測るか

`app_session_torch._gen_once` のブロック再構築構造
`block = [SP(pooler over kept)] + [recall(evicted blocks)] + [raw window(last rw=512)]`
を最小再現し、**想起クエリの出どころだけ**を 3 アームで切り替える:

| arm | 想起クエリ | 相当 |
|---|---|---|
| `OFF` | なし(`rec_emb=None`) | SP 圧縮のみ。具体値は失われるはず(対照) |
| `TURN` | ユーザ質問、ターン頭で 1 回 | 現行 BLOCKRECALL_V1 / 本番 |
| `MID` | 各再構築で現在の CoT 末尾を採り直す | **COTSCAN 本題(per-position)** |

想起本体は `block_recall.BlockArchive`(学習不要・モデル QK 空間の pre-RoPE クエリ ×
ブロックキー)。pooler は本番と同じ `AttnPoolSP`(`fft_out/pooler.pt`)。
判定 = 露出窓外(evicted)に植えた厳密値が最終回答に出るか + 正しいブロックが
想起トップに来たか。

バッテリーは direct(質問が値を名指す)と deferred(CoT が数ステップ進んでから値が要る)
の混在。deferred は per-position が効くはずの想定ケース。

## 実行方法(GPU)

`boot.sh` が RunPod 等の GPU ポッド上で HF からコード/重みを取得 → `fft_hf` を構築 →
`cotscan_live.py` を実行 → 結果を HF にアップロード → 自己終了する。CPU でも
`OUT_DIR=out python cotscan_live.py`(`fft_hf/` と `fft_out/pooler.pt` を隣に置く)で動くが、
R1 長考 CoT は CPU では非常に遅い。

成果物は HF の `trigger_experiment/results_cotscan_live/`(results.json / live.log)。

## 結果

(実行後に追記)
