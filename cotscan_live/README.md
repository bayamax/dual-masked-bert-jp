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

## 結果(A5000, fp32 1.5B, seed=0, 実生成 CoT, 各 ~124s)

`results_v2.json` 参照。指標は 3 つに分離した:
- **operand_recovered** = SP で露出窓外に追い出した *植えた値* が、生成出力(think含む)に
  戻ってきたか。**「想起が効いたか」の本指標。**
- **task_correct** = 最終回答が正しいか(direct=値そのもの、deferred=計算結果)。
- **gold_block_in_topk** = 想起が注入した top-k(=2)に gold ブロックが入っていたか。

| arm | operand_recovered | task_correct | gold_block_in_topk(計測可分のみ) |
|---|---|---|---|
| **OFF**(想起なし) | **0/4** | 0/4 | 0/4 |
| **TURN**(現行・ターン頭1回) | **3/4** | 3/4 | 2/2(direct) |
| **MID**(COTSCAN・per-position) | **4/4** | 3/4 | 2/2(direct) |

### 結論

- **核心の問いは Yes。** SP 圧縮だけ(OFF)では植えた値は **0/4** で全滅 = 圧縮が高エントロピー値を
  失うという前提を実生成 CoT 上で再現。想起を入れると値が戻る。**教師強制でなく実 CoT で成立**
  (COTSCAN 残作業 #2 に回答)。
- **per-position 配線(残作業 #1)が機能。** 各再構築で CoT 末尾をクエリに引き直す MID が
  operand を **4/4** 回復。TURN(ターン頭1回)が落とした `price_direct` も MID は拾う。
  MID の唯一の task ミス(`ref_code_direct`)は operand は think 内に回復済みで、最終回答へ
  書き出されなかった出力整形の取りこぼし(=想起の失敗ではない)。
- **想起ランキングは生 QK では弱い。** gold ブロックは一貫して **rank-2**(filler ブロックが
  rank-1 を取る)で、top-k=2 注入に救われて回復している。これは COTSCAN_SPEC の
  「生 qk は弱く、学習インデクサ(Phase B)で rank-1 が安定する」という記述と整合。
  本番化では `phaseb_indexer.npz` を射影に使えば rank-1 化が見込める。
- 単問バッテリーでは MID と TURN の operand 回復差は 1 問(4 vs 3)で、spec の
  「単問では MID≈TURN、真価は多段 CoT」という予測とも整合。

### 再現

GPU ポッドで `boot.sh`(HF からコード/重み取得 → `fft_hf` 構築 → 本スクリプト実行 →
結果を HF `trigger_experiment/results_cotscan_live_v2/` にアップロード → 自己終了)。
v1(計測粗・top-1 比較)は `trigger_experiment/results_cotscan_live/`、本確定版 v2 は
`..._v2/` に保存済み。

---

## GSM8K(本数を増やした検証)— `cotscan_gsm8k.py`

4 シナリオは少なすぎるという指摘を受け、**GSM8K test の問題文(数値ごと)を
filler で rw=512 の外へ追い出してから「上の問題を段階的に解け」**と指示する、
多段 CoT の本番寄りバッテリーで再検証(N=25、greedy、`results_gsm8k_v1.json`)。

| arm | 正答 | 平均 recall 回数 |
|---|---|---|
| **OFF**(想起なし) | **0/25 (0%)** | 0 |
| **TURN**(現行・ターン頭1回) | **9/25 (36%)** | 1 |
| **MID**(naïve per-position・置換) | **4/25 (16%)** | 9 |

paired: TURN✓MID✗ = 7、MID✓TURN✗ = 2、both = 2、neither = 14。

### 発見(スモークと逆)— naïve per-position は**悪化する**

- OFF 0/25:数値が evicted されると解けない(floor 確認)。recall は必須。
- **MID < TURN**。MID の誤答は `1`, `2`, `99999999999999991611392`, `1234567890` 等の
  **崩壊**で、平均 recall 9 回。原因は実装が **毎再構築で recall を“置換”** していた点:
  CoT 末尾が問題文から語彙的に離れると、想起が別ブロック(filler 等)に切り替わり、
  **まだ必要な問題文ブロックを外す** → 数値を失って degeneration。
  TURN は 1 回ピン留めして固定するので安定。
- 教訓:COTSCAN の per-position は「必要なブロックを**足す**」ものであって、
  「毎回**入れ替える**」ものではない。置換実装は多段 math で有害。

### `MID_ACC`(蓄積=ピン留め)は効かなかった — 仮説は外れ

各再構築で top-k ブロックの**スコアを累積**し上位の**和**を注入(問題文ブロックを
ピン留めする狙い)。だが結果は変わらず:

| arm | 正答(`results_gsm8k_v2.json`) |
|---|---|
| TURN | 9/25 (36%) |
| MID(置換) | 4/25 (16%) |
| **MID_ACC**(蓄積) | **4/25 (16%)** |

MID と MID_ACC が同値 → **「置換 vs 蓄積」は本質ではない**。両者の共通点は
**「毎再構築で CoT 末尾をクエリに再想起する」**こと。TURN との違いは 2 軸あり、
(1) クエリ源(指示文 vs CoT 末尾)、(2) 再注入頻度(1 回 vs 毎回)。

### アブレーション `MID_Q`(クエリ=指示文・再注入=毎回)→ 犯人が確定

| arm | 正答(`results_gsm8k_v3.json`, N=24) |
|---|---|
| TURN(query=指示文・**1回固定**) | 8/24 (33%) |
| MID(query=CoT末尾・毎回) | 4/24 (17%) |
| MID_Q(query=指示文・毎回) | **4/24 (17%)** |

**MID と MID_Q は 24 問中 23 問で予測が完全一致。** クエリ源(質問文 vs CoT 末尾)は
結果に**影響しない** → 犯人は **クエリではなく「生成中に recall を毎再構築で差し替えること
自体」**。正しいブロックを入れても、64 トークンごとに注入文脈が変わると 1.5B の推論が
揺れ、greedy が脱線して `1` `2` `1234567890` 等に崩壊する。

## 総合結論

| 観点 | 結論 |
|---|---|
| 想起の要否 | **必須**。OFF は GSM8K 0/25(数値が SP-evict されると 1.5B は解けない) |
| どの配線が良いか | **ターン頭で 1 回だけ呼び出して固定(TURN=現行 BLOCKRECALL/Addendum)が最良(33–36%)** |
| naïve per-position | **有害**。置換/蓄積/クエリ源によらず 16–17% に悪化 |
| なぜ | 教師強制・単一事実(COTSCAN_SPEC)では「引き戻し位置の隠れ表現で top-2=1.0」だが、
**実生成・多段 CoT では注入文脈を動かし続けること自体が小モデルを不安定化**する |

`block_recall` の QK 想起(= full-KV ならアテンションが飛んだ先を当てて生トークンを戻す)
という**仕組み自体は機能する**(小スモーク・TURN で実証)。だが COTSCAN_SPEC 残作業 #1 の
「per-position を生成ループで毎ステップ」を**素朴に**配線すると実 CoT では逆効果。
忠実な per-position 化には、(a) トリガーゲートで**真の発火位置だけ**に限定し、
(b) 注入は**一度差したら固定**(毎ステップ差し替えない)等、安定化が要る。これが
COTSCAN_SPEC 自身の「実 CoT では発火位置が変わりうるので生成ループ実装後に再検証」という
留保の、具体的な中身。

### 全成果物(HF `trigger_experiment/`)
- `results_cotscan_live{,_v2}/` — 4 シナリオ小スモーク(v2 が確定版)
- `results_gsm8k_v1/` — OFF/TURN/MID(N=25)
- `results_gsm8k_v2/` — TURN/MID/MID_ACC(N=25)
- `results_gsm8k_v3/` — TURN/MID/MID_Q アブレーション(N=24)
