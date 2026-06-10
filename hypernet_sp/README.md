# HyperNet-SP: STATUS 2026-06-10 の推奨次ステップ実装

`hypernet-sp-distill`(HF)の STATUS_2026-06-10.md「Recommended Next Steps」3件の実装と検証。
ファイルは HF リポジトリ直下にそのままコピーして使う(`runtime/` 依存なし、各ファイル自己完結)。

| ファイル | 行き先 (HF repo) | 役割 |
|---|---|---|
| `intent_route.py` | `./` | 次ステップ3: 信頼度の3バンド条件化ルーティング |
| `eval_intent_routing.py` | `evals/` | 新旧ルーティングの 5-fold CV A/B 評価(本環境で実行済み) |
| `test_intent_route.py` | `evals/` | 単体テスト(BGE/sklearn 不要、13/13 pass) |
| `anaphora.py` | `./` | 次ステップ2: pin エンティティの Web クエリ付加 |
| `test_anaphora.py` | `evals/` | 単体テスト(純 stdlib、13/13 pass) |
| `attn_export3.py` | `./` | 次ステップ1: attention プローブを 30 シナリオに拡張(STEP1, MLX) |
| `attn_probe3.py` | `./` | 次ステップ1: LOO-CV ロジスティック・トリガー学習(STEP2, torch) |

## 1. Intent ルーティングの条件化(実測済み)

**問題**(STATUS): top-1 確率 < 0.5 で crude regex に全面降格し、`I/my → recall` が
「any tips for my weekend trip to Kyoto?」型の質問を誤って個人記憶 recall に送る。

**実装**: 3バンド判定。`p1 >= 0.5` は従来通り分類器、`0.30 <= p1 < 0.5` のグレーゾーンは
**分類器の top-2 候補内に限定**して構文キューで裁定(top-2 の外へは跳ばない。gold が top-2 に
入る率は実測 93.8%)、`p1 < 0.30` のみ従来 regex。加えて「疑問文 fact」の補正を一律 recall
ではなくキュー分岐(math / recall / chitchat / lookup)に変更 — これが
「how do I get from Kyoto station to Kinkaku-ji?」(clf=fact, p=0.52)を recall に送っていた真因。

**結果**(intent_train.py のコーパス 837 例 + intent_gen.jsonl、本物の BGE-small で fold 内学習の
honest CV):

| 指標 | 旧 (p>=0.5 else regex) | 新 (3バンド) |
|---|---|---|
| 全体 5-fold CV | 797/837 = **0.952** | 807/837 = **0.964** |
| グレーゾーンのみ (81例, 9.7%) | 57/81 = **0.704** | 67/81 = **0.827** |
| Kyoto チャレンジ12例 (STATUS の失敗モード) | **6/12** | **12/12** |

統合(`tiered_rag_mlx.py`):
```python
from intent_route import route_intent
def intent_of(text, min_conf=0.5):
    return route_intent(text, _intent_clf(), _bge(), hi=min_conf)
```

## 2. 照応解決(Web クエリへの pin エンティティ付加)

**問題**(STATUS): "what are the best temples to visit there?" が文脈なしで DuckDuckGo に
渡り大阪の寺が返る。

**実装**: lookup クエリが照応表現(there/it/she/the place …)を含み、かつ自前のアンカー
エンティティ(文中の固有名詞・コード)を持たないときだけ、pins → session の新しい順に
エンティティを抽出して `query (Kyoto)` 形式で付加。代名詞の書き換えはしない(検索エンジンには
追加キーワードで十分)。訂正は最新優先(runtime の "corrections win" 規約と一致)。
`scorer` フックで BGE 関連度ランキングへ差し替え可能。

統合(`tiered_rag_mlx.py` の lookup 分岐と `TieredMemory.retrieve` の L3 直前):
```python
from anaphora import expand_web_query
name, ch = _web_retrieve(expand_web_query(user_msg, self.mem.pins, self.mem.session))
```

## 3. Attention-mass トリガー(30 シナリオ + リーク無し CV)

`attn_export2/probe2.py` の規約を踏襲して 30 シナリオ(math 10 / code 6 / name 6 / time 4 /
attr 2 / meas 2 — トリガーが算数文専用にならないようカテゴリを分散)。`attn_probe3.py` は:

- **leave-one-scenario-out CV**(ref/ctrl ペアを一緒に hold out)
- **ヘッド選択(top-6 delta)を各 fold 内で再実行** → 報告 AUC に選択リークなし
- fold 横断のヘッド選択安定性レポート(単一タスク族でしか勝てないヘッドの検出)
- 全データで fit した `attn_trigger3.joblib` {heads, clf, mu, sd} を保存

実行は Mac 側(MLX + `fft_hf/` が必要): `python3 attn_export3.py && python3 attn_probe3.py`。
CV ロジック自体は合成データで検証済み(植え込んだ 3 ヘッドを正しく回収、AUC 0.983、pairwise 30/30)。

ランタイム接続(probe が有望なら): ターン開始時に eager-attention の 1 forward で選択ヘッドの
SP 質量を読み、トリガー確率が閾値を超えたら外部検索(L1/L2/L3)を発火 — 「参照型か?」の
ヒューリスティック判定を、モデル自身の参照シグナルで置き換える。

## 再現

```bash
python3 test_intent_route.py                       # 13/13, 依存なし
python3 test_anaphora.py                           # 13/13, 依存なし
python3 eval_intent_routing.py --hf-repo <path>    # 要 torch/transformers/sklearn + BGE DL
```
