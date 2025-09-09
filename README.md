# Dual-Masked BERT 日本語学習

本リポジトリでは、BERT エンコーダ出力である文脈ベクトルを直接マスク予測し、その後マスクトークンを復元する 2 段学習を一気通貫で行う Dual-Masked BERT を実装します。

## 1. 構成
```
AI dev/
├── requirements.txt
├── README.md
├── src/
│   ├── model.py
│   ├── data.py
│   └── __init__.py
└── train.py
```

## 2. セットアップ
```bash
# ローカル
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. vast.ai での学習手順
1. vast.ai ダッシュボードで新規インスタンスを作成し、PyTorch / CUDA イメージを選択。
2. 「Bind Git Repo」に本リポジトリ URL を設定。`--branch main` など指定可能。
3. 起動後、下記コマンドで学習開始。
```bash
cd /workspace/AI dev
python train.py \
  --model_name_or_path cl-tohoku/bert-base-japanese \
  --dataset_name wikipedia \
  --dataset_config_name 20230501.ja \
  --output_dir outputs \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3
```

## 4. 主要スクリプト
- `src/model.py` : DualMaskedBERT 実装
- `src/data.py`  : マスク処理含むデータセット
- `train.py`     : 学習スクリプト
