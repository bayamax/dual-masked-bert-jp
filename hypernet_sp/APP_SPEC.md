# APP_SPEC — hypernet-sp-distill アプリ開発仕様書

これ1枚でアプリ開発に着手できることを目的とした仕様書。
詳細な根拠・実測値は `RELEASE_REPORT.md`、移植手順は `HANDOFF_PORTING.md`(いずれも同梱)。

---

## 0. 正典宣言(Source of Truth)— 最初に読むこと

**モデルの正確な運用方法(あるべき挙動)は、本リポジトリ `hypernet_sp/` ディレクトリが唯一の正典である。**

| 場所 | 状態 | 扱い |
|---|---|---|
| **`hypernet_sp/app_session_torch.py`** | **検証済み13修正が全て適用された turn() の正典** | 運用ロジックはこのファイルの挙動が正。迷ったらここを読む |
| `hypernet_sp/*.py`(memory_core 等6モジュール) | 正典の構成部品(純Python、テスト付き) | そのまま流用する |
| リポジトリ直下の `tiered_rag_mlx.py` `sp_mlx.py` 等 | **レガシー**(修正前の旧本番ランタイム) | 挙動仕様として読まないこと。MLX 移植の「土台」としてのみ使い、`HANDOFF_PORTING.md` §2 のチェックリストで修正を当てる |
| 手元のローカルクローン | 本リポジトリ main より新しいことはあり得ない | 食い違ったら必ず HF main に揃える |

「どっちが新しいか」で迷ったら: 各修正はコミットメッセージに欠陥番号と実測根拠つきで
記録されている。`RELEASE_REPORT.md` の欠陥表(#1〜#13)と `app_session_torch.py` の
コメントが一致しているものが最新。

---

## 1. プロダクト定義

**有界メモリで動くオンデバイス推論チャットアプリ。**
1.5B 推論モデル(DeepSeek-R1-Distill-Qwen-1.5B / FFT 済み)が、会話履歴を
32 本のソフトプロンプト(SP)に圧縮しながら、階層メモリ(L1 セッション / L2 ディスク /
L3 Web)と連携して動く。KV キャッシュは会話がどれだけ長くなっても **O(1)**。

- ターゲット: Apple Silicon(MLX)。iPhone は 3GB メモリ予算で成立(実測 1.6–1.8GB)
- UI 言語: 日本語(翻訳シム経由 — §6)。内部処理・メモリ・モデルは全て英語

## 2. ロードする成果物(全て本リポジトリ内)

| 成果物 | パス | 用途 |
|---|---|---|
| 学生モデル(FFT済) | `fft_out/student.pt` → `build_fft_hf.py` で HF 形式化 / 4bit は `fft_mlx4` | 本体 |
| SP pooler(AttnPoolSP 32×1536) | `fft_out/pooler.pt` | 履歴→SP 圧縮+eviction 質量 |
| intent ルータ(6クラス) | `hypernet_sp/intent_head.npz` | ターン種別判定 |
| specificity probe | `hypernet_sp/specificity_head.npz` | 固有値 pin 判定 |
| 検索トリガー(任意・**v1 は OFF**) | `hypernet_sp/attn_trigger4_head.npz` | SP 参照検知→メモリ再注入(v3 は使用禁止) |
| 文埋め込み | BAAI/bge-small-en-v1.5(33M) | 検索・信頼度 |
| ランタイムロジック | `hypernet_sp/*.py`(純Python 6 モジュール) | §3 のパイプライン |

npz ヘッドは `export_classifiers.LinearHead`(numpy のみ)でロード。sklearn 不要。

## 3. ターンパイプライン仕様(契約)

入力: `user_msg: str` / 出力: `(answer: str, source: str|None, chunks: list[str])`
リファレンス実装: `hypernet_sp/app_session_torch.py` の `AppSession.turn()`(全修正適用済み)。

```
1. 保存依頼検知   wants_persist+値あり → 保存して即 ack(疑問形でも)
2. intent 判定    intent_route.route_intent(3バンド: 高確信/top-2裁定/regex)
3. fact          → ログ+即 ack(生成しない)
4. pin           recall/lookup 以外で固有値があれば pin(上限12・ローテーション)
5. recall        → 統合検索(L1+L2 を一括ランク+訂正連鎖)
                   ・ヒットなし → 正直 miss(生成しない。捏造防止)
                   ・信頼度<0.62 → 最近接メモを逐語提示(断定しない)
                   ・≥0.62 → 隔離コンテキストで逐語引用(_clean_quote)
6. lookup        → known-fact-first(意味≥0.6+内容語共有)→ WM pin → Web
                   Web 前に照応解決(expand_web_query)、後に injection ガード
7. math/command  → 文脈注入: 疑問形除外+関連度フィルタ+(outdated)/(current)タグ
                   +前回結果の平叙文注入。テンプレは質問先頭・(a)(b)(c)検出でパート指示
8. 生成          SP-evict ループ(rw=512–1024, maxD=4096, C=64)
                   ・compute 系(math/command): think 強制+DecodePolicy(二相温度、
                     収束/ループ強制終了)+2パス救済
                   ・それ以外(雑談・作文・要約等): **空 think 先行で直接回答**(temp 0.6)
                     — think は math 装置。創作で強制すると成果物が think 置き去りになる(#20)
9. 後処理        groundedness 検査+リトライ(≤2)、電卓検算(評価エラー修復)、
                   compute 結果の自己記録(旧結果は置換)
```

## 4. 設定値(検証済み・変更時は再検証のこと)

| 項目 | 値 | 備考 |
|---|---|---|
| temp / 最終回答 | 0.6 / ~greedy | 二相デコード |
| rw(raw window) | 512–1024 | 品質ダイヤル。KV は rw=1024 でも ~30MB |
| maxD(eviction) | 4096 | 質量ベース間引き。142 回連続発火まで検証済み |
| ターン上限 | 2000 tok / 240s | 多部回答は ANSCAP を可変に(#13) |
| recall 確信閾値 | 0.62 | 未満は最近接メモ提示 |
| known-fact 閾値 | 意味 0.6 + 語共有 | シャドーイング防止の二重ロック |
| 検索トリガー | **v1 は OFF**(任意機能) | 後日 ON 時は attn_trigger4 / th=0.6(MULTITURN_TRIGGER_RESPONSE 参照) |
| pin 上限 / 検索 cap | 12 / 4 | |

## 5. できること(全て実測済み)

- **無限長会話で一定メモリ・一定速度**: 10.9k tok 生成・eviction 142 回で速度平坦(4.5→4.0 tok/s 相当)、合計フットプリント 1.6–1.8GB
- **長期記憶**: 40 ターン後でも序盤の事実を逐語リコール(マラソン 36/36)。訂正は最新が勝つ(2 回訂正・元の値に戻るケース含む)。セッション跨ぎ永続(L2)
- **正直さ**: 未保存の値は捏造せず「保存されていない」(0 トークン)。不確かな検索ヒットは断定せず最近接メモを提示
- **推論**: 単発 GSM8K 級 80–90%、self-contained math は Web を引かない、計算結果の参照チェーン(合計→再計算→お釣り)
- **Web 連携**: 照応解決(「there」→ Kyoto 付加)、injection ガード、groundedness 検査つき
- **複合テスト実績**: v1 12/12・v2 17/17・v3 18/18 相当・v4 36/36・v5 23/29(残は §6 の天井)

## 5.5 会話の継続性(2026-06-11 追加・検証済み)

雑談を含む全ターンが直前までの会話の流れを踏まえる(v6 バッテリー 8/8、回帰 v1 12/12・v2 17/17)。

- **ターン縫い込み(必須)**: 生成しないターン(fact 即 ack・recall 引用・正直 miss・
  最近接メモ提示)も `User/Assistant` 交換をトークンとしてストリームに記録
  (`AppSession._stitch`)。これが無いと「京都行ってきた」→ack の後の
  「ハイライト何だった?」が繋がらない
- **意図対応の救済(#14)**: think が cap 到達した時の強制クローズ継続は、compute 系のみ
  「Final answer: 」(48 tok)、雑談/説明はそのまま自然回答(200 tok)。
  「Final answer:」固定だと説明が数字スタブ化する(実測: binary search の説明→「100」)
- **収束強制終了は compute 系のみ武装**(逐語ループ検出は全ターン有効)
- (任意)セッション跨ぎ: SP ウォームスタート(`save_state`/`load_state` — 前セッションを
  SP に圧縮した状態で再開)+プロフィール前置(evict されない MQ 領域に「ユーザーについて」
  1 行)。アプリを開き直しても続きから話したい場合に有効化

## 6. 制限(設計上の前提として組み込むこと)

1. **逐語の事実は SP に置けない**(アーキテクチャの大前提)。コード・数値・固有名詞は
   raw window か検索で扱う — 本パイプラインは自動でそうするが、独自改変時は厳守
2. **モデル天井(1.5B)**: 多部問題の帳簿管理(全計算が正しくても違う値を選ぶ —
   実測で算術 25/26 正解なのに最終値を誤った)。対策はサブ質問分割(アプリ側で
   (a)(b)(c) を個別ターン化)。算術の評価エラー自体は電卓検算が修復
3. **一般指示はモデルに通じない**(「最新値を使え」等)。制御は機械的タグ・
   サーバー側解決・テンプレ構造で行う(本パイプラインは適用済み)
4. **多言語**: 内部は英語のみ。日本語 UI は翻訳シム(`translation_shim.py`、
   逐語スパン保護つき)経由。実 NMT でのプレースホルダ生存確認が未了
5. **検証の範囲**: 全て単一シード・CPU。実機で複数シード再走と
   レイテンシ/発熱計測が必要(マラソンバッテリーは無修正で再利用可)
6. **セキュリティ**: web_guard はパターン第一線。恒久対策(span 制約デコード)は未実装

## 6.5 既知の移植バグ対応(Swift SP-evict 崩壊 — 2026-06-10 回答済み)

アプリ側報告(`GENONCE_DEGENERATION_REPORT.md`)の `,1!#!!!!` 型崩壊は**検証済みで、
モデル/レシピの問題ではない**(torch リファレンスは同一重み・同一プロンプト・ガード無効で
200 トークンの整合出力)。Swift 移植の忠実性バグであり、対応手順は以下を参照:

| 参照先 | 内容 |
|---|---|
| `hypernet_sp/GENONCE_DEGENERATION_RESPONSE.md` | 判定・原因候補の再ランク・照合手順 |
| `hypernet_sp/parity_reference.npz` | 機械照合用の基準データ(下記) |
| `hypernet_sp/sp_evict_parity.py` | 基準データの再生成スクリプト |

**最有力原因**: pooler の cross-attention に「past が空ならスキップ」のガード
(`past.size(1) > 0`)が無い場合、空キー集合への softmax が NaN を返し SP 全体が
NaN になる。初回ターンは rw=512 の間ずっと past が空なので、自明なプロンプトで
即座に再現する — 報告症状と完全整合。

**照合手順(この順で)**: ① Swift 側で空入力 pooler の出力に NaN が無いか確認
② 全 32 ベクトルの L2 ノルムが **1.468531**(|out_scale|)に一致するか
③ cache crop 後の RoPE 位置が MQ から再開しているか(`parity_reference.npz` の
`q_ids` は 14 個 → 位置 14 から)④ `greedy_tokens` と突き合わせて最初の分岐点を特定。

## 6.6 ブロック引き戻し(Block Recall)— 2026-06-12 正式採用

セッション内の逐語記憶層。露出窓から SP へ追い出されたトークンを 128-tok ブロックの
まま保持し(SP はこれまで通り雰囲気の継承)、毎ターン BGE cosine で top-2 ブロックを
[SP] と [露出窓] の間に**逐語再注入**する。実測: 正確値想起 0/10→7/10、
フルKV(4k で 1/10 — 小型モデルは全文が窓内でも文中の値を拾えない)に勝つ。
回帰ゼロ(v1 12/12、v2 は ON/OFF 完全一致)。

移植契約(Swift):
1. 追い出しトークン列を 128-tok ブロックに封緘し、decode したテキストの BGE
   ベクトル(384 次元)を 1 本付与。未封緘の端数(<=127 tok)も検索時は擬似ブロック
   として採点(封緘境界ギャップ — CPU 実測でニードル取りこぼしの原因)
2. ターン頭にユーザ文を BGE クエリ化 → 全ブロックと cosine → top-2 を時系列順に
   連結し、ブロックのトークン id をそのまま [SP] の直後に並べる(テンプレ装飾なし)
3. リファレンス実装: `hypernet_sp/block_recall.py`(mode="bge")+
   `app_session_torch.py` の `archive` 配線(検索は吸収処理の後、1 ターン 1 回)
4. フラグ運用: 環境変数 `SPCHAT_BLOCK_RECALL=bge`。OFF なら従来動作と完全一致

## 7. 開発開始手順

1. 本リポジトリを clone、`HANDOFF_PORTING.md` の順で `tiered_rag_mlx.py` に移植(0.5–1 日)
2. `composite_test1–5` を MLX ラッパで再走(期待値は HANDOFF §6)
3. 実機計測(レイテンシ・発熱・複数シード)
4. 翻訳シム結合(Apple Translation framework 推奨 — 予算 0 byte)
5. 計装(引用メモリ ID・groundedness 通過・再質問シグナルのオプトインログ)を
   **初日から**仕込む — メモリバンク構想の燃料

## 8. ファイルマップ(hypernet_sp/)

実装: `memory_core.py` `intent_route.py` `anaphora.py` `web_guard.py` `decode_policy.py`
`calculator.py` `translation_shim.py` `app_session_torch.py`(リファレンス)
ヘッド: `intent_head.npz` `specificity_head.npz` `attn_trigger3_head.npz`
テスト: `test_*.py`(8 スイート・120+ assertion・MLX 不要)
バッテリー: `composite_test{,2,3,4,5}.py` / 評価: `eval_sp_value.py` `eval_intent_routing.py`
文書: `RELEASE_REPORT.md`(判定+欠陥13件の記録)`HANDOFF_PORTING.md`(移植手順)
