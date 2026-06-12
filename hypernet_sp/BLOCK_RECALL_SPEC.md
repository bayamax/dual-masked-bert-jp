# Block Recall 実装仕様書(A方式)— アプリ実装用

2026-06-12 正式採用。これ 1 枚で実装に入れることを意図した自己完結仕様。
リファレンス実装: `hypernet_sp/block_recall.py`(mode="bge")と
`hypernet_sp/app_session_torch.py` の `archive` 配線。疑問が出たらコードが正。

## 0. なにを解決するか(実測値つき)

SP(32 ベクトル圧縮)は「何の話をしていたか」は保持するが**正確な値を落とす**。
4213 トークン履歴に埋めた 10 個の事実(パスワード・内線番号・重量など)の想起実測:

| 構成 | 正解 |
|---|---|
| SP のみ(現行) | 0/10 |
| **SP + Block Recall(本仕様)** | **7/10** |
| フル KV(全履歴を注意窓に入れる) | 1/10 |

フル KV が 1/10 である点が本質: **1.5B は 4k 級になると、窓内にあっても文中の値を
拾えない**(lost-in-the-middle)。関連ブロックを質問の直前に再注入する方が強い。
回帰: 既存バッテリー v1 12/12、v2 は recall ON/OFF で結果完全一致(差分ゼロ)。

## 1. データ構造

```
Block:
  ids:  [Int]          # 追い出されたトークン id、ちょうど BLOCK=128 個(順序保持)
  emb:  [Float; 384]   # decode したテキストの BGE-small 文書ベクトル(L2 正規化済み)
Archive:
  blocks: [Block]      # 時系列順、append-only
  buf:    [Int]        # まだ 128 に満たない端数(最新の追い出し分)
```

メモリ: 1 ブロック ≈ 128×4B(id)+384×4B(vec) ≈ 2KB。1 万トークンの履歴でも ~160KB。
永続化はセッション内のみで良い(v1)。`save_state` の kept 列から再構築可能。

## 2. 書き込みパス(トークンが露出窓から出る時)

生成ループの「absorb」(露出窓 rw=512 から溢れたトークンを kept へ移す処理)と
**同じ場所・同じトークン列**で `archive.buf` に append する。
`len(buf) >= 128` になるたび先頭 128 個を封緘:

1. `ids = buf[:128]; buf = buf[128:]`
2. `text = tokenizer.decode(ids)`(チャットマーカー `<｜User｜>` 等は**そのまま含めて良い**)
3. `emb = BGE.encode_document(text)`(is_query=False、L2 正規化)
4. blocks.append(Block(ids, emb))

SP 側の処理(pooler への kept 供給、maxD eviction)は**一切変更しない**。

## 3. 読み出しパス(毎ターン 1 回)

タイミング: ターンの生成開始時、**absorb 処理の直後**(その瞬間の最新ブロックまで
検索対象に入る)。1 ターンに 1 回だけ検索し、結果はそのターンの全 rebuild で再利用。

1. `q = BGE.encode_query(user_message)`
   ※ **BGE のクエリ側プレフィックス必須**:
   `"Represent this sentence for searching relevant passages: " + user_message`
2. 候補 = blocks 全部 + (len(buf) >= 16 なら buf を擬似ブロック化して追加 —
   **封緘境界ギャップ**: 窓から出た直後の <=127 tok はここに居る。これを入れないと
   直近に追い出された値が取りこぼされる。実測で発見済みの必須項目)
3. `score_i = dot(q, emb_i)`(両者正規化済みなので cosine)
4. top-k(**k=2**)を選び、**時系列順に**並べ替えて ids を連結 → `rec_ids`(<=256 tok)

## 4. 注入(コンテキスト組み立て)

毎 rebuild のブロック構成を次の順にする(現行との差分は rec の挿入のみ):

```
[MQ プライム(profile)] [SP(32 vec)] [rec_ids の埋め込み] [露出窓 rw トークン] [feed]
```

- rec_ids は**トークン埋め込みとして逐語注入**。プロンプトテンプレートで包まない・
  引用符をつけない・「Recalled:」等のラベルもつけない(ストリームの過去断片が
  そのまま戻ってきた形が最も自然に効く)
- kept が空(=SP なし)のターンでも rec があれば注入してよい(欠陥 #16 の
  「空 SP を入れない」規則は SP にのみ適用)

## 5. 既存パイプラインとの関係(変更しないもの)

- intent ルーティング、L1/L2 メモリ、closest-note、honest miss、電卓、DecodePolicy:
  **全て無変更**。Block Recall は「生成コンテキスト側」の層で、メモリ層と独立
- 露出窓 rw=512、C=64、maxD=4096、SP 32 vec: 無変更
- フラグ: リファレンスでは環境変数 `SPCHAT_BLOCK_RECALL=bge` で ON。
  **OFF 時は従来動作と bit 単位で同一**(切り戻し自由)。アプリ側も同等の
  feature flag を用意すること

## 6. パリティ検証手順(移植後に必ず)

1. `hypernet_sp/needle_recall_test.py` の `build_history`/`NEEDLES` と同一の
   履歴・質問を Swift 側に流す(`--conds sp,bge` 相当)
2. **検索一致**: 各質問で選ばれた top-2 ブロック index が torch 実装と一致すること
   (BGE が同一重み・同一プレフィックスなら一致する。ズレたらまずクエリ
   プレフィックス、次に正規化、次に decode 文字列を疑う)
3. **エンドツーエンド**: bge 条件 7/10 ± 1(サンプリング揺れ)、sp 条件 0/10 近辺
4. 回帰: 既存のアプリ動作確認スイートを recall ON で 1 周(期待: 差分ゼロ)

## 7. 既知の限界と次フェーズ

- 7/10 の取りこぼし 3 件は「正しいブロックが top-2 に入らない」検索ミス。
  k を上げると窓を圧迫するので k=2 を推奨値とする
- Phase B(学習インデクサ、`phaseb_*.py`)が検索精度の改善を担当(進行中)。
  採用された場合も**本仕様のスコアリング関数が置き換わるだけ**で、
  データ構造・注入・配線は変わらない設計
- 多セッション永続(ブロックのディスク保存と世代管理)は v2 スコープ

## 8. B方式(アンサンブル)実装仕様 — 2026-06-12 追記

A方式(§1-6)に**スコアリングの差し替えだけ**を加える。データ構造・注入・配線は不変。
実測: needle 10/10(A=7/10)、未知テンプレ検索 top-2 .983(A=.894)。参照実装
`ensemble_archive.py`、学習済み重み `hypernet_sp/phaseb/phaseb_indexer.npz`(数百KB)。

### スコア計算(retrieve 時)

score_i = z(bge_i) + 1.5 * z(idx_i)   (z = スコア列の平均0/分散1正規化、top-2選択)

- bge_i: §3 と同じ BGE cosine
- idx_i: 学習インデクサ。npz の Aq[L,Hq,D,d], Bk[L,Hkv,2D,d], w[L,Hq](meta=[3,12,2,128,32]):
  q'=einsum(q,Aq)、k'=einsum(kfeat,Bk) を GQA 6 倍複製、sim=ReLU(q'·k')、
  score=Σ softmax(w)·sim(層・ヘッド重み付き和)。numpy のみで実装可(Swift は Accelerate)

### 特徴量(ここが B の本体・パリティの急所)

1. **ブロックキー kfeat[nB,L,Hkv,2D]**: 層 (8,14,20) の k_proj **出力(pre-RoPE)**を、
   ブロックが追い出される前に**文脈付き**で取得し、ブロック内 mean と max を連結。
   文脈付き=そのトークンが生成 forward を通った時の k_proj 値。アプリでは生成中の
   rebuild forward にフックして収穫するか(推奨・無料)、検証時は履歴全体を 1 回
   forward して取る(参照実装はこちら。学習データと同形)
2. **クエリ q[L,Hq,D]**: 質問を `<eos><User>{q}<Assistant><think>\n\n</think>\n\n` の形にし、
   **履歴キャッシュを背後に置いた状態で** forward した q_proj 出力の平均。
   ⚠ 素の単独 forward だと深層でドリフトし実測で 2/10 落ちる(§8 検証で確認すること)

### 検証手順(移植後)

1. needle ベンチ ens 条件 10/10 ± 1(`needle_recall_test.py --conds ens` と同一入力)
2. 中間値パリティ: 同一履歴で idx_i スコア列が torch 実装と一致(<1e-3)
3. 留意: 本番の逐次キー収穫形での再学習(V3)は未了。それまでは検証時も
   参照実装と同じ「履歴 1 回 forward」収穫で評価すること
