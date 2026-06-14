# RECALL_SPEC — ゲート式リコール生成 仕様書 (RECALL_V1, 2026-06-14)

SP 圧縮(露出窓 `rw` の外を pooler で gist 化)された無限ターン文脈において、**CoT/生成の
各位置で「窓の外へ落ちた事実が今いるか」を検知(GATE)し、必要なら正しい evicted ブロックを
想起(RETRIEVE)して逐語再注入する**機構の仕様・学習法・検証結果・成果物をまとめる。
COTSCAN_SPEC.md / APP_SPEC_ADDENDUM.md(`hypernet_sp/`)の後継・実装確定版。

ベース: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`(FFT 版 = `fft_hf`)+ SP pooler(AttnPoolSP)。

---

## 1. 機構(3部品)

| 部品 | 役割 | 入力 | 出力 |
|---|---|---|---|
| **RecallGate** | いつ引くか | 現在位置の pre-RoPE クエリ(層8/14/20, 4608次元) | 発火スコア(>閾値で発火) |
| **Retriever** | どれを引くか | 同クエリ + evicted ブロック表現 | top-k ブロック |
| **RecallRuntime** | 配線 | モデル+pooler+上記 | `prepare(seq)`→特徴 / `decide(ctx,p)`→発火&想起 |

Retriever は 3 系統:
- **QKIndexer**(学習): モデル自身の QK 空間。各ブロックの pre-RoPE キー要約(~3KB/block)が要る。
- **BGERetriever**(学習・状態レス): **保持は token ids のみ**。発火時に候補ブロックを BGE で逐次
  エンコードし、固定の小ヘッド(~数十万パラ)で隠れクエリと橋渡し。無限ターンでも増えるのは ids だけ。
- **raw-QK**(学習なし、`block_recall.BlockArchive`): モデルの QK 内積で逐語ブロックをスコア。

---

## 2. 学習方法(本体 SFT なし)

**データ**: 実 Dolphin-R1 の長 CoT(≥1000 トークン)。長考なので CoT 前半が窓外へ自然にスクロール
アウトし、「引き戻しが必要な瞬間」が大量に生じる(レアイベントを母数で稼ぐ)。

**ラベル(教師=実アテンション)**: フルKV で各窓位置のアテンションを測り、**窓(512)より前の
evicted 領域へ集中して飛んだ位置=引き戻し位置(正例)**、飛んだ先のブロック=gold。
層8/14/20・ヘッド和・BOSシンク除外・行正規化。

**特徴**: 同位置の **SP 圧縮レイアウト** `[BOS][SP(evicted)][窓512]` での pre-RoPE クエリ(本番で
実際に得られる隠れ表現)。

**学習**: GATE=ロジスティック回帰(4608次元)。Indexer=DSA lightning-indexer 型(~20万パラ)を
KL でアテンション分布に一致させる。BGEヘッド=隠れクエリ↔BGE埋め込みの小MLP。
item 分離 heldout で評価。

> 注: 学習部品(Indexer/Gate)は **Dolphin ドメインに in-domain**。別ドメイン(例 GSM8K の
> 質問位置クエリ)には分布外で、その場合は学習なし raw-QK の方が転移する。

---

## 3. 検証結果

### 3.1 検知・想起の汎化(実 Dolphin・item分離 heldout, 正例 ~3,951)
- **GATE 検知 AUC = 0.9584**
- **想起 top-2**:学習QK **0.9995**(raw-QK 0.807)/ **学習BGEヘッド 0.9997**(raw-BGE cosine 0.286)
- → 「BGE に学習変換をかませば壁(~0.8)を破れる」= 実証。**状態レス(ids のみ)構成が QK保存版と同等**。

### 3.2 単体テスト(`recall_kit` ロード→fixtures、全 PASS)
gate 分離 AUC 0.94 / QK top-2 1.000 / BGEヘッド top-2 1.000 / BGE encode OK。

### 3.3 複合テスト(配線 end-to-end、未見held 46問・2117位置)
- 想起@真の引き戻し:QK **0.999** / BGE **0.979**
- end-to-end(発火して正しいブロック):QK **1.0** / BGE **0.969**
- GATE: precision 0.75 / recall 0.48(出荷閾値=TPR優先。下げれば recall↑、best-F1点で R 0.79)

### 3.4 SP ランタイム自体のタスク影響(GSM8K, filler なし, 同採点)
| 条件 | 正答率 | 補足 |
|---|---|---|
| full-KV(無圧縮) | **66.7%** | 基準 |
| SP @ rw=1024 | **66.7%** | CoT が窓に収まり圧縮 0 → full-KV と同一 |
| SP @ rw=512 | 60.0% | 19/30 が実圧縮 → SP圧縮の実コスト ~7pt |

→ **SP 圧縮は優秀**(rw=1024 で無損失、rw=512 でも-7pt)。GSM8K 規模では recall を持ち出すまでもない。

### 3.5 圧縮が効く長尺(Dolphin 長CoT, CoT>rw, n=5)
| arm | 正答率 |
|---|---|
| FULL | 3/5 (60%) |
| SP@rw1024 | 2/5 (40%) |
| SP_RECALL | 2/5 (40%) |

- 長尺では **圧縮が実コスト(60→40)**。recall は n=5 で差し引きゼロ(1救済・1破壊)。
- **trace で「ちゃんと引けている」ことを確認**:発火は毎回「元問題を読み直そうとした瞬間」に起き、
  窓外へ落ちた**元の問題文を正確に引き戻していた**(例: "check the original problem again"→発火
  score 13.04→two-trains 問題を想起→`\boxed{440/9}` 正解)。
- 正答率が動きにくいのは**引きの失敗ではなく 1.5B の解答力がボトルネック**(難問は手元にあっても
  解けない)。

---

## 4. 成果物(HF `baya1116/hypernet-sp-distill`)

**`recall_runtime/`(3.87GB・自己完結バンドル, これ一式で即動く)**
```
fft_hf/                ベースLLM(HF形式, from_pretrained 可)
pooler.pt              SP pooler
components/            gate.npz / indexer.npz / bge_head.npz(学習済み重み)
recall_kit/            RecallGate / QKIndexer / BGERetriever / RecallRuntime
deps/                  attn_export3_torch(pooler loader)/ block_recall / attn_scenarios
README.md              ロード&実行例
```
検証ログ・各実験結果: `trigger_experiment/`(results_dolphin* / recall_kit_v4 / results_gsm_* 等)。
コード: GitHub `bayamax/dual-masked-bert-jp` ブランチ `claude/hypernet-sp-spec-review-qrafn9`。

### ロード例
```python
import torch, sys; sys.path += ["deps", "."]
from transformers import AutoModelForCausalLM, AutoTokenizer
from attn_export3_torch import load_pooler
from recall_kit import RecallGate, QKIndexer, RecallRuntime
tok = AutoTokenizer.from_pretrained("fft_hf")
llm = AutoModelForCausalLM.from_pretrained("fft_hf", torch_dtype=torch.float32).eval().cuda()
pooler = load_pooler("pooler.pt").cuda().eval()
dims = (llm.config.num_attention_heads, llm.config.num_key_value_heads,
        llm.config.hidden_size // llm.config.num_attention_heads)
gate = RecallGate.load("components/gate.npz")
qk   = QKIndexer.load("components/indexer.npz", dims, device="cuda")
rt = RecallRuntime(llm, tok, pooler, gate, qk=qk, device="cuda")
ctx = rt.prepare(seq_ids)
for p in range(ctx.win_len):
    d = rt.decide(ctx, p, backend="qk")     # gate → (発火時のみ) 想起
    if d["fired"]:
        inject(ctx.block_ids[d["blocks"][0]])
```

---

## 5. 知見・限界

1. **SP圧縮は強い**: rw=1024 では full-KV と無差。recall の真価は「CoT/会話が rw を大きく超える長尺」。
2. **recall は正しい中身を正しい瞬間に引ける**(質的に実証)。最終正答への寄与はモデルの解答力次第で、
   短サンプル×難問では見えにくい。
3. **早期注入(ターン頭)> mid-CoT 発火**(解くだけなら早い方が良い)。ゲートの価値は「無駄打ち削減」。
4. **filler で人工的に窓外へ追い出す評価は不適切**(窓が filler で埋まり評価を歪める)。自然な長尺で測る。
5. **ドメイン外注意**: 学習 Indexer/Gate は Dolphin in-domain。別分布では raw-QK 等の学習なし手法が転移しやすい。

## 6. 残課題
- 1.5B でも解ける難度の長尺問題で母数を増やし、recall のタスク寄与を有意に測る。
- 発火時の巻き戻し(rewind)・閾値の運用最適化(recall↑)、MLPゲート(AUC 0.950→0.961)の採用。
- 再注入フォーマットの改善(逐語ブロック→問題文の完全復元等)で長尺の圧縮損失(60→40)を埋める。
- `RecallRuntime` の本番 / MLX 移植(`HANDOFF_PORTING.md` のパリティ手順)。

---

## 7. 全成果物の所在(ありか)一覧

HF リポジトリ: **`baya1116/hypernet-sp-distill`**(URL は `https://huggingface.co/baya1116/hypernet-sp-distill/tree/main/<path>`)
git: **`bayamax/dual-masked-bert-jp`** ブランチ **`claude/hypernet-sp-spec-review-qrafn9`**

### 7.1 モデル一式(これ一式で即動く)
| 物 | 所在 |
|---|---|
| 完全バンドル | HF `recall_runtime/` |
| ベースLLM(HF形式) | HF `recall_runtime/fft_hf/` |
| SP pooler | HF `recall_runtime/pooler.pt` |
| 検知ゲート / QK想起 / BGE想起 重み | HF `recall_runtime/components/{gate.npz, indexer.npz, bge_head.npz}` |
| パッケージ | HF `recall_runtime/recall_kit/` |
| 依存 | HF `recall_runtime/deps/` |
| 本書 | HF `recall_runtime/RECALL_SPEC.md` |

### 7.2 学習済み重みの原本+テスト
| 物 | 所在 |
|---|---|
| gate / indexer / bge_head + 単体・複合レポート + fixtures | HF `trigger_experiment/recall_kit_v4/artifacts/` |
| パッケージ src(同上の出所) | HF `trigger_experiment/recall_kit_v4/src/` |
| ベース重み(原本) | HF `fft_out/student.pt`(3.55GB), `fft_out/pooler.pt`(302MB) |

### 7.3 検証結果(主要)
| 検証 | 所在 |
|---|---|
| 検知AUC 0.958 + 学習QK top-2 0.9995(Dolphin heldout) | HF `trigger_experiment/results_dolphin/` |
| BGE比較(raw 0.286 → 学習 0.9997) | HF `trigger_experiment/results_dolphin_bge/` |
| 探索(ゲート動作点 / MLP 0.961 / 後方スキャン) | HF `trigger_experiment/results_explore/` |
| full-KV 66.7% | HF `trigger_experiment/results_gsm_fullkv/` |
| SP@rw512 60% / SP@rw1024 66.7% | HF `trigger_experiment/results_gsm_sp512/`, `results_gsm_sp1024b/` |
| 長尺 FULL/SP/SP_RECALL(60/40/40) | HF `trigger_experiment/results_dolphin_solveB/` |
| 引き戻しの中身(trace) | HF `trigger_experiment/results_dolphin_trace/` |
| 新ゲート×raw-QK の GSM8K(OFF0/TURN31/GATE17) | HF `trigger_experiment/results_gsm_gate/` |
| 初期GSM8K知見(素朴per-positionは有害) | HF `trigger_experiment/results_gsm8k_v1..v4/`(git: `cotscan_live/results_gsm8k_v*.json`) |

> 補助・中間ラン(`results_gsm_new*`, `results_gsm_rw1024`, `results_cotscan_live*` 等)も
> 同 `trigger_experiment/` 配下に残置(filler 等のアーティファクト確認用)。

### 7.4 コード(git ブランチ)
| 物 | 所在(git) |
|---|---|
| パッケージ + 本書 | `recall_kit/`(gate.py / retriever.py / runtime.py / __init__.py / README.md / RECALL_SPEC.md) |
| 実験スクリプト一式 | `cotscan_live/`(dolphin_scan.py 学習 / build_package.py / composite_test.py / explore.py / gsm8k_sp.py / gsm8k_fullkv.py / cotscan_gsm8k.py / dolphin_solve.py 等)+ 各 boot_*.sh + 結果 JSON |

### 7.5 元仕様・前提(本書の出発点)
| 物 | 所在 |
|---|---|
| COTSCAN / Addendum 仕様 | HF `hypernet_sp/COTSCAN_SPEC.md`, `hypernet_sp/APP_SPEC_ADDENDUM.md` |
| Phase B 想起インデクサ(既存) | HF `hypernet_sp/phaseb/phaseb_indexer.npz` ほか |
| COTSCAN 実験コード(自然CoT等) | git ブランチ `claude/huggingface-model-status-ypclnu` の `trigger_experiment/`(cot_recall_natural.py 等)+ `RESULTS.md` |
