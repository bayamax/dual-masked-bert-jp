# RESPONSE: SP-evict Degeneration (MLX-Swift port) — 判定と切り分け手順

`GENONCE_DEGENERATION_REPORT.md` への回答。依頼された決定的テストを実施した。

## 判定: レシピ/重みはクリーン。欠陥は Swift 移植の忠実性にある

依頼通り、**PyTorch リファレンス `_gen_once` を同一重み・同一プロンプト
("What is 47 multiplied by 9?")・greedy・リピートガード/ポリシー全て無効**で実行した
(`sp_evict_parity.py`、本ディレクトリ)。結果:

- 出力は 200 トークンにわたり完全に整合的な英語推論(`,1!#!!!!` 型ノイズの兆候ゼロ)
- 空 past に対する SP(後述)も NaN なし・全 32 ベクトルのノルムが |out_scale| に一致

加えて Python-MLX 実装は既に PyTorch と max abs diff 2.4e-7 で検証済み(RESULTS.md)。
つまり **torch ✓ / Python-MLX ✓ / Swift ✗** — 変数は Swift 移植のみ。

## 原因候補の再ランク(レポートの候補に診断を加味)

### 1.(最有力・新規)空キー集合に対する softmax = NaN

**「自明なプロンプトでも壊れる」が最大の手がかり。** 初回ターンでは kept バッファが空で、
しかも rw=512 のため**最初の 512 生成トークンの間、全リビルドが「長さ 0 の past」への
pooler 呼び出し**になる。リファレンス実装(torch / Python-MLX とも)は

```python
if past is not None and past.size(1) > 0:   # ← このガード
    a, w = self.cross(...)
```

で cross-attention を**スキップ**する。Swift 移植でこのガードが欠けると、
長さ 0 のキーに対する softmax が NaN を返し、SP 全体が NaN → 直後から全出力が崩壊する。
症状(シード不問・自明プロンプトで常時再現・SP 経路のみ)と完全に整合。

**確認方法**: Swift 側で `pooler(空入力)` の出力を `parity_reference.npz` の `sp0` と比較。
NaN チェック → 全 32 ベクトルの L2 ノルム = **1.468531**(|out_scale|)であること。

### 2. cache crop 後の RoPE 位置(レポートの候補 1)

Python-MLX は `c.trim(c.offset - MQ)` で offset=MQ に戻し、注入ブロックの RoPE 位置は
MQ から続く。Swift 側で trim 後の offset が 0 に戻っていると位置が巻き戻り崩壊する。
**確認方法**: 注入ブロック先頭トークンの位置 index が MQ(本テストでは 14)であること。

### 3. SP の dtype / スケール(レポートの候補 2)

pooler は fp32 で計算し、**モデル dtype(4bit 実行時は fp16)へキャストしてから**注入する。
ノルムのアンカーは上記 1.468531。

### 4. 質量プーリング / eviction(レポートの候補 3)→ このバグでは無関係

自明プロンプトでは kept が空で eviction は構造上発火しない。候補から除外してよい
(直る前に触らないこと)。

### 5. 4bit との相互作用(レポートの候補 4)→ 可能性低

Python-MLX 4bit は同経路で 36 tok/s の整合生成実績あり(RESULTS.md)。

## パリティ照合手順(`parity_reference.npz`)

| キー | 内容 | 照合 |
|---|---|---|
| `q_ids` | プロンプトのトークン列(チャットラッパ+`<think>\n` 込み、14 個) | 完全一致(ここがズレてたら以降全部ズレる) |
| `sp0` | 空 past の SP [32×1536] fp32 | NaN なし、近似一致(fp16 なら ~1e-3) |
| `sp0_norms` | 各ベクトルの L2 ノルム | 全て 1.468531 ±1e-4 |
| `block0_checksum` | 最初の注入ブロックの総和 | 近似一致 |
| `greedy_tokens` | greedy 200 トークンの参照列 | **最初の分岐点**を特定する用途。4bit では数十トークン以降の完全一致は期待しない(数値誤差で分岐し得る)。最初の 1 トークン目から違う場合は上記 1〜3 のどれか |

照合順は表の上から。`sp0` が NaN ならその時点で原因 1 が確定する。

## 再現コマンド(参照側)

```bash
# fft_hf/ と fft_out/pooler.pt の隣で
python3 sp_evict_parity.py     # 判定+ parity_reference.npz を再生成
```
