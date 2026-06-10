# アプリ公開判定レポート (2026-06-10)

対象: hypernet-sp-distill(R1-Distill-Qwen-1.5B + AttnPoolSP + tiered RAG)。
本レポートの数値は全て本セッションの Linux/CPU 環境で、HF リポジトリの実重み
(`fft_out/pooler.pt`, `student.pt` → `fft_hf`)・実分類器・実データで計測した。
再現コマンドは各節に記載。プロジェクトの長期ビジョン(オンデバイス展開+使用フィードバック
によるメモリバンク成長)を判定基準に織り込んだ。

## 判定サマリ

| 領域 | 判定 | 根拠 |
|---|---|---|
| SP 圧縮の価値 | **PASS** | drop に 16/20、プラセボに 18/20 勝利(下記) |
| intent ルーティング | **PASS**(対策後) | CV 0.964、グレーゾーン 0.827、Kyoto 型 12/12 |
| 検索トリガー | **PASS** | LOO-CV AUC 0.970、運用点 recall 0.97 / FPR 0.10 |
| メモリ層の堅牢性 | **PASS**(対策後) | 20/20 + 14/14(敵対入力・JP・injection) |
| 日本語対応 | **CONDITIONAL** | 語彙層は修復済み。encoder/intent は英語のみ(下記) |
| 生成の分散・終端 | 実装済み・要実機計測 | decode_policy + スモーク(下記) |
| モバイル移植性 | **PASS**(対策後) | 全分類ヘッドを numpy/npz 化(sklearn 依存除去) |
| レイテンシ/発熱 | **未計測** | 実機(M 系/iPhone)でのみ測定可能 — 残項目参照 |

---

## 1. SP 圧縮の価値(コア品質ゲート)

**問い**: SP は「過去 think を捨てる」(R1 公式テンプレート挙動)より本当にマシか。
held-out の dolphin-r1 reasoning(1500–3500 tok — 学習フィルタ ≥4000 と構成上排反)20 件、
同一予測ターゲット(96 tok)に対する 4 条件の CE。`eval_sp_value.py --n 20`(distant=512, rw=128)。

| 条件 | mean CE | 備考 |
|---|---|---|
| full attention | 0.5242 | 上限(無制限 KV) |
| **SP + recent(出荷構成)** | **0.6180** | |
| recent のみ(drop) | 0.6413 | R1 公式テンプレート相当 |
| ノイズ SP + recent(プラセボ) | 0.6482 | 「32 本の余白」効果の統制 |

- **SP < drop: 16/20、SP < プラセボ: 18/20** — SP は実情報を運んでいる(枠の効果ではない)
- full とのギャップ +0.094 は学習時測定(~0.09)と独立再現
- 回復率は drop→full ギャップの **19.9%** — SP は「上限回復」ではなく「削除の緩和」。
  逐語事実を raw window / 検索で扱う現行アーキテクチャの妥当性を裏づける

**含意**: スマホ(3GB、SP 必須)でも品質根拠が立つ。同時に、SP の改善余地
(層別 KV 注入など)は大きい — 公開後の伸びしろ。

## 2. Intent ルーティング(STATUS 問題 → 対策済み)

`eval_intent_routing.py`。コーパス 837 例、fold 内学習の honest 5-fold CV。

| 指標 | 旧 | 新(3 バンド) |
|---|---|---|
| 全体 | 0.952 | **0.964** |
| グレーゾーン(9.7%) | 0.704 | **0.827** |
| Kyoto 型チャレンジ | 6/12 | **12/12** |

真因は閾値以下の regex 降格に加え、「疑問文 fact → 一律 recall」補正だった(p=0.52 の
自信あり誤分類も recall 行き)。キュー分岐に変更。

## 3. Attention-mass 検索トリガー

30 シナリオ × 6 タスク族、選択リークなし LOO-CV: **AUC 0.970、pairwise 30/30**。
運用表(`trigger_runtime.py`、in-sample のため閾値選定用):

| th | recall | FPR | precision |
|---|---|---|---|
| 0.3–0.4 | **0.97** | 0.10 | 0.91 |
| 0.5 | 0.87 | 0.03 | 0.96 |
| 0.6 | 0.80 | 0.00 | 1.00 |

コスト非対称(取りこぼし=数値の confabulation、誤発火=検索 1 回)より **th=0.35 推奨**。
CPU fp32 全層 eager で 1.3s/call → Metal では ms オーダー。選択 6 ヘッドは全て L≤22。

## 4. メモリ層の堅牢性(発見 → 対策実装済み)

`test_memory_robustness.py` 20/20、`test_web_guard.py` 14/14。

**発見 1 — 日本語の語彙検索が完全に死んでいた**(`_words` が ASCII 専用 → JP クエリの
トークン集合が空)。→ **CJK 文字バイグラムで修復**(`memory_core.py`)。

**発見 2 — bge-small-EN は日本語で精度崩壊**: pos/neg マージン +0.07〜0.14(英語 +0.34)、
かつ絶対類似度 0.65–0.78 が EN 校正の min_sim=0.46 / top_delta=0.12 を素通り →
**意味層が日本語で distractor を注入する**。→ CJK クエリは語彙優先 + 意味層は
高閾値(0.72/0.06)フォールバックに変更。恒久対策は多言語 encoder
(multilingual-e5-small 等)への移行だが、intent/specificity ヘッドの再学習を伴うため公開後。

**発見 3 — groundedness は injection を素通しする**(salient トークンが文脈にあれば
命令文でも合格)。→ `web_guard.py`: L3 チャンクの命令パターン(EN/JP)+ zero-width
文字除去。パターンベースの第一線であり完全ではない(恒久は span 制約デコード)。

**注**: intent 分類器・specificity probe・decode_policy の正規表現も英語前提。日本語 UI で
公開するなら、ルータの JP 学習データ追加が必要(コーパス生成パイプラインは既存)。

## 5. デコード方策(分散・終端対策)

`decode_policy.py`(単体 14/14): (1) 二相温度 — think 内 0.6 / `</think>` 後 ~greedy
(multi-turn 2–4/5 の分散源は最終文のサンプリング)、(2) 答え振動検出 — 同一正準値が
k=3 回主張されたら `</think>` 強制(長さでなく収束で発火する budget forcing。
「1157 を出し続けて最後に 1000」型の自己崩壊を直接遮断)。
生成スモーク(`smoke_gen.py`、本物の SP-evict ループ、n=3 × 2 アーム、cap 600、同一シード):

| アーム | 正答 | 総トークン | 自然終端 |
|---|---|---|---|
| base(常時 temp 0.6) | 3/3 | 1800 | **0/3**(全問 cap 到達) |
| policy | 3/3 | 1581 (−12%) | 1/3(振動停止発火 → 381 tok で正答終端) |

base は 3 問とも一度も `</think>` に到達せず cap まで再考し続けた — STATUS の over-thinking
そのもの。policy は discount 問で「0.8×25=20」を 3 回主張した時点で発火し、`Final answer: 25`
を出して EOS。**発見と追対策**: 振動検出は「値の収束ループ」は捉えるが「問題文の再読ループ」
(natalia)は発火しない → 第二トリガー(think 内 6-gram × 4 の逐語ルート検出)を実装済み。
ただし逐語でない**パラフレーズ再読**はなお素通りする — 残る over-thinking は per-turn cap が
最終防衛線のまま(現状維持で可、改善は実機での think 長分布を見てから)。
**統計的検証は実機の gsm8k_eval に policy を載せて行うこと**(CPU では n を張れない)。

## 6. モバイル移植性

- 出荷 joblib は sklearn 1.7.2 のピクルで、1.9.0 で既に InconsistentVersionWarning。
  iOS(mlx-swift)に sklearn は存在しない。→ **全 3 ヘッド(intent/specificity/trigger)を
  係数 npz + numpy 専用スコアラへエクスポート、sklearn と一致 <1e-18 を検証**
  (`export_classifiers.py`、`*_head.npz` 同梱)
- メモリ予算(3GB 端末): 重み 4bit ~1GB + pooler ~45MB + BGE ~70MB + KV(rw=1024)~30MB
  + ランタイム ~0.5GB ≈ **1.6–1.8GB で成立**。4GB RAM 端末(アプリ上限 ~2GB)は要実機判断

## 7. 複合テスト(アプリ使用想定の統合バッテリー) — 最終 12/12 PASS

`composite_test.py` + `app_session_torch.py`(MLXランタイムの決定単位の torch 移植)。
3 セッション・12 チェック: 永続化→プロセス再起動→L2 リコール、distractor 汚染と記録衛生、
L1 リコール、自己完結 math、**pin 依存フォローアップ math**、L3 lookup(injection ページ混入)、
照応解決(there→Kyoto 拡張)。

**1回目 9/12 → 原因究明と修正(計装つき再現 5 回)→ 最終 12/12。** 発見した欠陥チェーン:

| # | 欠陥 | 真因 | 修正 |
|---|---|---|---|
| 1 | fact 発話後の雑談に前ターンの値が混入 | fact 発話を全文生成していた(仕様は即 ack) | fact→ログ+即 ack 短絡(各ターン 0s 化) |
| 2 | L2 リコールに無関係な L1 が同乗 | L1/L2 を別々にランキング(union でない) | ストア横断プールで min_sim/top_delta 適用 |
| 3a | フォローアップ math が前の答えを返す | 収束検出が中間値で発火(「gets $26 back」語順を動詞リストが見落とし) | 正規表現拡充+競合値ガード+最新主張一致 |
| 3b | (同上・継続) | 注入文脈に前の質問だけがあり答えがない→全再導出で混乱 | 計算結果の自己記録 |
| 3c | (同上・継続) | 長い meta 指示テンプレートを 1.5B が「質問」として解釈 | 質問先頭・短い括弧書きテンプレートに簡素化 |
| 3d | (同上・決定打) | **疑問形テキストを文脈注入するとモデルがそれを再回答する** | 自己記録の平叙文化+疑問形 pin の注入除外 |

修正後の該当ターン: 24→26 とも正答、469 トークンでクリーン終端(修正前は 600 cap 到達)。

**本番への含意**: 3c/3d のテンプレートと「疑問形を注入しない」原則は `tiered_rag_mlx.py`
から逐語移植した部分で見つかったので、**実機の本番システムに同じ欠陥が存在する**。
GSM8K multiturn の失敗(6/10)の一部はこれが原因である可能性が高く、同じ修正の移植を推奨。

### 複合バッテリー v2(敵対的パターン) — 最終 17/17 PASS

v1 が触れなかった使用パターン: 訂正の上書き、複数事実の同時リコール、英字コード逐語、
個人事実による世界知識のシャドーイング、**未保存情報のリコール(捏造耐性)**、
command 再計算を挟む 3 段連鎖 math、distractor 後の深いリコール。
**1回目 14/17 → 原因特定・修正 → 17/17。** 新たに発見・修正した欠陥:

| # | 欠陥 | 真因 | 修正 |
|---|---|---|---|
| 4 | 訂正後の色が回答に反映されない | 照応的で短い訂正文は multi-fact クエリと類似度が低く(実測 0.541 vs hi 0.744)、**top_delta が正当な訂正を distractor として落とす** | 訂正連鎖: 選択チャンクの後にあり訂正マーカー+内容語共有する行を強制同乗(fixpoint まで連鎖) |
| 5 | 未保存の wifi パスワードを聞くと **"password123" を捏造** | recall で検索ゼロ→自由生成に落ちる経路 | 空 recall は生成せず正直 miss を即答(0 トークン) |
| 6 | 連鎖 math で「24 なら 26、32 なら…」と両論ヘッジ | 自己記録の結果行が複数残り、現在値が判別不能 | 結果行はワーキング状態として扱い、再計算で置換 |

#4 の「top_delta が訂正を殺す」は `tiered_rag_mlx._sem_matches` 由来 — **本番にも存在する**。
#5 は本番の `turn()` にも同じ自由生成フォールスルーがあり、要移植。

複合テスト通算: v1 12/12 + v2 17/17(いずれも修正後のクリーン通し)。

注意(誠実さのため): 各バッテリーは単一シード・1 通し。「ある程度大丈夫」の水準であり、
統計的保証ではない。実機では複数シードで multi-turn バッテリーを回すこと。

## 8. 残項目(この環境では測定不能 — 公開前に実機で)

1. **レイテンシ/発熱/バッテリー**(M 系 Mac・iPhone 実機): 特に CoT 4000 tok ターンの
   サーマルスロットリング。decode_policy の振動停止が実機でどれだけ平均 think 長を削るか
2. **gsm8k_eval n≥100 + policy on/off**(MLX、実機): スモークの方向性確認を統計に
3. **多言語 encoder 移行**(e5-small 系)+ intent の JP データ追加
4. **メモリバンク向け計装**(ビジョン直結・公開初日から必須): 引用メモリ ID / groundedness
   通過 / 再質問シグナルのローカルログ。スキーマは別途設計可
5. web_guard はパターン第一線 — span 制約デコード or NLI 検証を恒久対策として

## 成果物一覧(本ブランチ `hypernet_sp/`)

実装: `intent_route.py` `anaphora.py` `web_guard.py` `decode_policy.py` `memory_core.py`
`trigger_runtime.py` `export_classifiers.py`
評価: `eval_intent_routing.py` `eval_sp_value.py` `smoke_gen.py` `attn_export3(_torch).py`
`attn_probe3.py`
テスト(計 87 assertion、全て pass): `test_intent_route.py` `test_anaphora.py`
`test_memory_robustness.py` `test_web_guard.py` `test_decode_policy.py`
アーティファクト: `attn_trigger3.joblib` `attn_trigger3_head.npz` `intent_head.npz`
`specificity_head.npz` `attn_probe3.npz` `sp_value_results.json`
