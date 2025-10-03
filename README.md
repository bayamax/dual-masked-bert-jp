## LLaMA 内的表現 → トークン埋め込み 変換と再投入（Mac対応）

このリポジトリは、LLaMA系の小規模オープンモデル（デフォルト: TinyLlama 1.1B）を用いて、
単一トークン入力時の内的表現（隠れ状態）とトークン埋め込みのペアを収集し、
内的表現からトークン埋め込みへ写像する小さな変換モデルを学習、
さらに得られたトークン様ベクトルを `inputs_embeds` として再投入するデモまでを提供します。

### 特徴
- Mac(MPS)対応（Apple Silicon 推奨）。
- モデル: TinyLlama/TinyLlama-1.1B（変更可）。
- 内的表現の抽出層を指定可能（最終層がデフォルト）。
- 線形/MLPのシンプルな変換モデルで学習（MSE損失）。
- 変換結果の評価（cos/近傍トークン一致率）。
- 再帰的圧縮トークン投入デモ（旧文を1ベクトルに圧縮し先頭へ付与）。

---

## 1. セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # zsh/bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

デフォルトモデルは `TinyLlama/TinyLlama-1.1B-Chat-v1.0` です。初回実行時に自動でダウンロードされます。

Apple Silicon (MPS) を利用可能な場合、自動で `mps` デバイスを使用します。CPU でも動作します。

---

## 2. 内的表現とトークン埋め込みのペア抽出

単一トークンを（必要に応じてBOSと共に）モデルへ入力し、
対象トークン位置の隠れ状態（内的表現）と、そのトークンの入力埋め込みをペアとして収集します。

```bash
python -m src.extract_pairs \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --layer_index -1 \
  --max_tokens 20000 \
  --batch_size 256 \
  --output_path data/pairs.pt
```

主な引数:
- `--layer_index`: 取り出す層（`-1` は最終層）。
- `--max_tokens`: 収集する語彙数（省略で全語彙）。
- `--batch_size`: 抽出バッチサイズ。
- `--bos`: BOS を前置するか（デフォルト有効）。

出力 `data/pairs.pt` には以下が保存されます：
- `X`: 内的表現テンソル [N, d_model]
- `Y`: トークン入力埋め込みテンソル [N, d_model]
- `token_ids`: 対応するトークンID [N]
- メタ情報（`model_name`, `layer_index` など）

---

## 3. 変換モデルの学習（内的表現 → トークン埋め込み）

```bash
python -m src.train_converter \
  --dataset_path data/pairs.pt \
  --model_type mlp \
  --hidden_dim 2048 \
  --epochs 10 \
  --batch_size 1024 \
  --lr 1e-3 \
  --save_path artifacts/converter.pt
```

`model_type` は `linear` か `mlp` を選択可能です。`mlp` の場合は中間次元 `--hidden_dim` を指定します。

---

## 4. 変換モデルの評価

```bash
python -m src.eval_converter \
  --dataset_path data/pairs.pt \
  --converter_path artifacts/converter.pt \
  --topk 5
```

出力指標:
- MSE / Cosine 相関
- 近傍検索による元トークン一致率（top-1 / top-k）

---

## 5. 再帰的圧縮トークン投入デモ

旧文（長いセンテンス）をモデルの隠れ状態に集約（平均等）し、
学習済み変換でトークン様ベクトルへ写像します。これを `inputs_embeds` として
プロンプト先頭に1トークン分相当で付与し、続く通常トークンと共に生成します。

```bash
python -m src.demo_recursive \
  --converter_path artifacts/converter.pt \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --compress_text "ここに旧文..." \
  --prompt "ここに現在の指示..." \
  --max_new_tokens 128
```

内部では `model.generate(inputs_embeds=...)` を使用し、
仮想トークン埋め込み + 通常トークン埋め込みを連結して推論します。

---

## 実装上のメモ

- 内的表現: 既定では「最終層の対象トークン位置の隠れ状態」を使用。
- トークン埋め込み: 入力埋め込み行列から該当トークン行を取得（多くのデコーダは入出力埋め込みがタイド）。
- 再投入: `inputs_embeds` へ直接埋め込みを与えることで“仮想トークン”を前置可能。
- MPS: `torch.backends.mps.is_available()` を見て自動選択。

### 参考: よくあるエラーと対処
- MPS上で半精度が不安定: `--dtype float32` で実行。
- generate時にpad_token_id未設定: 自動でtokenizer.pad_tokenを設定、うまくいかない場合は `--model_name` をTinyLlamaに。
- ダウンロード失敗: 一時的なネットワーク。`pip install -U transformers` 後に再試行。

---

## 既知の制約・注意

- 実験目的の簡易実装です。研究用途での検証に留めてください。
- 変換モデルは単純な写像であり、長文の完全圧縮・忠実な保持は保証できません。
- 一部モデルはライセンス上の同意が必要な場合があります。TinyLlama は制約が緩く扱いやすいです。

---

## ライセンス


---

## 6. ハイパーネットワーク型ソフトプロンプト生成（HyperPromptNet）

TinyLlama 1.1B を教師として、文脈隠れ状態から k 個の仮想トークン `e*` を直接生成するハイパーネットを学習します。教師の次トークン分布と整合するように KL を用いた蒸留を行い、必要に応じて中間層の隠れ状態やアテンション情報の一致を併用できます。

### 6.1 データセットの用意

`build_e_star_dataset.py` で、(text, follow_text) に対して最適化された `e*` と文脈隠れ状態を収集します。

```bash
python -m src.build_e_star_dataset \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --num_virtual 1 \
  --steps 200 \
  --hidden_weight 0.0 \
  --hidden_layer_index -2 \
  --hidden_loss_type cosine \
  --num_samples 50 \
  --save_path data/e_star_sup.pt
```

既に `data/` にはサンプルの集約/ストリーム済みデータがいくつか含まれています（`e_star_sup_agg_*.pt`, `e_star_sup_stream_*`）。

### 6.2 学習（教師あり蒸留）

`train_compressor_sup.py` は `--arch hyper` でハイパーネットを使用します。ベースの回帰損失（cosine+norm）に加えて、教師の次トークン分布に対する KL 蒸留、任意で隠れ状態一致/アテンション一致を足し合わせます。

```bash
python -m src.train_compressor_sup \
  --dataset_path data/e_star_sup.pt \
  --save_path artifacts/compressor_sup.pt \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --num_virtual 1 \
  --dtype float32 \
  --device cpu \
  --arch hyper \
  --num_heads 8 \
  --hyper_num_layers 2 \
  --hyper_ffn_dim 2048 \
  --kl_weight 0.2 \
  --kl_temperature 1.0 \
  --hidden_weight 0.0 \
  --attn_weight 0.0
```

オプション:
- `--attn_layers "-1,-2" --attn_use_weights --attn_weight 0.1` でアテンション特徴/重みの一致を利用
- `--hidden_weight 0.1 --hidden_layer_index -2 --hidden_loss_type cosine_norm` で中間層の一致を追加

### 6.3 評価（教師との一致とベクトル回帰）

学習済み圧縮器を、ベクトル回帰（MSE/Cos/RMSE_norm）と、教師の次トークン分布に対する CE/Top-k で評価します。

```bash
python -m src.eval_compressor_sup \
  --dataset_path data/e_star_sup.pt \
  --compressor_path artifacts/compressor_sup.pt \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --batch_size 16 \
  --topk 5 \
  --temperature 1.0
```

### 6.4 生成デモ（仮想トークン再投入）

`demo_compressor_sup.py` はテキストから文脈隠れ状態を抽出し、学習済み圧縮器で `e*` を生成してプロンプト先頭へ投入します。

```bash
python -m src.demo_compressor_sup \
  --compressor_path artifacts/compressor_sup.pt \
  --text "旧文の文脈..." \
  --prompt "現在の指示..." \
  --ctx_len 64 \
  --dtype float32 \
  --device cpu \
  --arch hyper \
  --hyper_num_layers 2 \
  --hyper_ffn_dim 2048 \
  --alpha 1.0
```

`--arch qattn` を指定すると、従来の単段 QueryAttention 圧縮器で実行できます。

各モデルのライセンスに従います。本リポジトリのコードは Apache-2.0 を想定します。


