# 本番移植 指南書 — hypernet_sp/ → tiered_rag_mlx.py (MLX)

引き継ぎ者向け。本ディレクトリの検証済み修正(欠陥 #1〜#13)とモジュール群を、
HF リポジトリの本番ランタイム `tiered_rag_mlx.py` / `sp_mlx.py` に移すための手順書。
背景・実測値・各欠陥の発見経緯は `RELEASE_REPORT.md` を先に読むこと。

## 0. 全体像 — 何がどこにあるか

`app_session_torch.py` は **`tiered_rag_mlx.ChatSession.turn()` の決定単位の写し**であり、
全修正が適用済みの「あるべき姿」のリファレンス実装である。迷ったら本番コードと
この写しを並べて diff を取ればよい。torch 固有なのはテンソル演算だけで、
**判断ロジック・プロンプト・閾値は 1:1 で移せる**。

| 移植元(本Dir) | 移植先 | 種別 |
|---|---|---|
| `memory_core.py` | `tiered_rag_mlx.py` の同名関数群を置換 | 純Python・そのままコピー可 |
| `intent_route.py` | `intent_of()` を置換 | 純Python・そのままコピー可 |
| `anaphora.py` / `web_guard.py` | 新規ファイルとして同居 | 純Python・そのままコピー可 |
| `decode_policy.py` | 新規ファイル+ `_gen_once` に配線 | 純Python+配線3行 |
| `app_session_torch.py` の `turn()` | `ChatSession.turn()` に判断ロジックを反映 | 手移植(下記 §2) |
| `*_head.npz` + `export_classifiers.LinearHead` | joblib ロードを置換 | sklearn 依存除去 |
| `attn_trigger3_head.npz` | 新機能(任意、§4) | |

**重要**: `memory_core.py` / `intent_route.py` / `anaphora.py` / `web_guard.py` /
`decode_policy.py` は **MLX を import しない**。コピーして import するだけで動く。
手作業が要るのは `ChatSession` クラス内部だけ。

## 1. 移植順序(依存関係順・各ステップでテスト可能)

1. **純Pythonモジュール5つをコピー**し、`tiered_rag_mlx.py` 内の重複定義
   (`_words`〜`TieredMemory`、`intent_of` と各 regex)を削除して import に差し替える。
   `_sem_matches`/`_match` は bge を引数で受ける形に変わっている点に注意
   (旧: グローバル `_bge()` / 新: 引数注入)。
   → **検証**: `test_memory_robustness.py` `test_intent_route.py` `test_anaphora.py`
   `test_web_guard.py` `test_decode_policy.py` をその場で実行(依存なしで回る)。
2. **`ChatSession.turn()` への判断ロジック移植**(§2 のチェックリスト)。
3. **`_gen_once` への DecodePolicy 配線**(§3)。
4. (任意)検索トリガー(§4)、電卓検証(§5、`calculator.py` がある場合)。
5. **検証**: composite バッテリー v1〜v4 を MLX で再走(§6)。

## 2. ChatSession.turn() チェックリスト(欠陥番号つき)

リファレンス: `app_session_torch.py` の `turn()`(~80行)。上から順に:

- [ ] **#疑問形保存依頼**: intent 判定の直後、
  `wants_persist + _FACTLIKE + specific_spans` なら persist+ack で即 return
- [ ] **#1 fact→即ack**: `intent == "fact"` は生成せず log+ack で即 return
  (本番は呼び出し側 app_demo が ack_only を渡しているが、turn() 内で強制する方が安全)
- [ ] **#5 空 recall の正直 miss**: recall で chunks が空なら生成せず
  「保存されていない」を即 return(自由生成に落とすと値を捏造する)
- [ ] **#10 不確信 recall は最近接メモ提示**: recall で chunks があっても
  BGE 信頼度 max-sim < **0.62** なら、断定せず最近接チャンクを逐語提示して return。
  ≥0.62 のみ従来の厳格引用テンプレートへ(プロンプト逃げ道も yes/no 判定も
  1.5B には効かないことを実測済み — 機械的提示が唯一動いた第3案)
- [ ] **#7 known-fact-first lookup**: lookup 分岐の先頭で
  `mem.retrieve_known(query)`(意味≥0.6 **かつ** 内容語共有の二重ロック)を照会。
  ヒットすれば引用パスへ。シャドーイング(emperor/Kyoto)は二重ロックが防ぐ
- [ ] **照応解決**: web へ行く直前に `expand_web_query(msg, pins, session)`
- [ ] **injection ガード**: web チャンクは `guard_chunks()` を通す
- [ ] **#3d 疑問形を文脈注入しない**: math/command 分岐の chunks 構築で、
  疑問形の pin / log 行を除外(`mc._is_question` でフィルタ)
- [ ] **pin の接頭辞重複排除**: log 行が pin を startswith で拡張している場合は pin を落とす
- [ ] **#3c+#12 計算テンプレート**: 質問先頭・短い括弧書き。
  `(a)(b)(c)` を検出したら「Answer EVERY lettered part…」、でなければ
  「End with the final number.」(単数指示は多部問題を途中完結させる)
- [ ] **#8 (outdated)/(current) タグ**: 注入前に `mark_superseded(rel)`。
  「最新値が正しい」という一般指示は 1.5B に通じない(2回実測)。タグは通じる
- [ ] **#4 訂正連鎖**: `_match` 経由なら自動適用(`_with_amendments`)
- [ ] **#3b+#6 結果の自己記録**: compute 成功後
  `"Earlier computed result: {val}."` を**平叙文で**記録し、**旧結果行は置換**
  (複数残すと両論ヘッジする)
- [ ] **#2+並び順**: `retrieve_personal` はストア統合プール
  (**persistent が先、session が後** — 逆だと古い L2 が新しい訂正に勝つ)

## 3. _gen_once への配線(3点)

1. **二相温度**: サンプリングの温度を `policy.temp(in_think, TEMP)` に。
   `</think>` 検出は直近 8 トークンの decode で行う(リファレンス実装の通り)
2. **収束/ループ強制クローズ**: 自由サンプリング中のみ
   `policy.note_text(現ターンのdecode)` が True なら
   `"\n</think>\n\nFinal answer: "` を feed に積む。
   ※競合値ガード(別候補が2回以上主張されていたら発火しない)込みの最新版を使うこと
3. **2パス救済**: 本番には既にある(逆に torch 側が未移植で v3 C7 を踏んだ)。確認のみ

## 4. 検索トリガー(任意・新機能)

`attn_trigger3_head.npz`(選択6ヘッド L1H0/L1H3/L21H4/L7H9/L22H9/L9H3 + 係数 + 標準化)。
ターン頭に eager attention 1 forward で各ヘッドの SP 質量を取り、`LinearHead` でスコア。
**th=0.35**(recall 0.97 / 誤発火 0.10)で外部検索を発火。
MLX 移植は選択ヘッドの QK だけ計算すれば軽い(全て L≤22)。LOO-CV AUC 0.970。

## 5. sklearn 依存の除去

`intent_head.npz` / `specificity_head.npz` を `export_classifiers.LinearHead`
(numpy のみ、sklearn と一致 <1e-18 検証済み)でロードする。joblib は
バージョンロックされたピクルであり、iOS には sklearn が無い。

## 6. 移植後の検証手順

1. 単体スイート(§1)— MLX 環境でそのまま回る
2. `composite_test*.py` は AppSession を import している。MLX 用には
   `AppSession(llm, tok, pooler, ...)` のコンストラクタ互換の薄いラッパを
   `ChatSession` に被せれば、**バッテリー本体は無修正で再利用できる**
   (これが torch 写しを作った最大の理由)。期待値: v1 12/12、v2 17/17、
   v3 18/18、v4 36/36。v5 はモデル天井によりほぼ 23–25/29
3. 実機では複数シードで v4 を 3 回以上(分散の確認 — CPU 検証は全て単一シード)

## 7. ハマりどころ(実際にハマった順)

- **ひらがな・漢字は Python regex の `\w`** — 日本語に隣接する英数に `\b` は効かない。
  lookaround を使う(`translation_shim.py` 参照)
- **採点・ログは ANSCAP 適用前の全文で** — 600 字 cap 後の文字列で何かを判断すると
  事故る(v5 で自己記録値が汚染され丸ごと1ラン無駄にした)。
  app 側も多部回答では cap を可変にすること(#13)
- 一般指示(「最新値を使え」「無ければ NOT FOUND と言え」)は **1.5B には効かない**。
  機械的タグ・機械的提示・サーバー側解決に置き換えるのが本プロジェクトの一貫した教訓
- バックグラウンドで cp→実行を連結する時は **バージョンマーカーをログで確認**
  (`HARNESS_V2` 方式)

## 8. 未移植のまま残るもの(意図的)

- 翻訳シム(`translation_shim.py`): 実 NMT との結合は実機作業。
  プレースホルダ `[[n]]` の生存確認を最初にやること
- メモリバンク計装: スキーマ未設計。公開初日から必要(RELEASE_REPORT §8)
- 多言語 encoder 移行: intent/specificity の再学習を伴うため公開後
