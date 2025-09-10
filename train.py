#!/usr/bin/env python
"""学習エントリポイント"""
import argparse
from pathlib import Path
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from tqdm.auto import tqdm

from src.model import DualMaskedBertForPreTraining
from src.data import DualMaskDataCollator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="cl-tohoku/bert-base-japanese-v2", help="Pretrained model (v2 uses SentencePiece and avoids MeCab dependencies)")
    parser.add_argument("--dataset_name", type=str, default="wiki40b", help="HuggingFace dataset name (default: wiki40b)")
    parser.add_argument("--dataset_config_name", type=str, default="ja", help="Dataset config name (default: ja for wiki40b)")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--map_batch_size", type=int, default=100, help="dataset.map 時の batch_size。大き過ぎるとメモリ不足や segfault の原因になる")
    parser.add_argument("--num_proc", type=int, default=1, help="dataset.map 時の並列プロセス数。SentencePiece はスレッドセーフでないため 1 を推奨")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Fast tokenizer (rust-tokenizers) を使うか。デフォルト False で Python 実装を使用し segfault を回避")
    parser.add_argument("--auto_shutdown", action="store_true", help="学習完了後にマシンを自動シャットダウン")
    return parser.parse_args()


def main():
    args = parse_args()
    # SentencePiece などの内部並列を制限
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=args.use_fast_tokenizer)
    # Fast tokenizer のスレッド並列を明示的に無効化（環境変数で設定できない場合の保険）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    # データセット読み込み
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split="train")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length)

    # batched=True で一定数ずつトークナイズする。バッチサイズとプロセス数を小さくすることで
    # SentencePiece 由来のセグフォやメモリ不足を回避する。
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
    )

    data_collator = DualMaskDataCollator(tokenizer, mlm_probability=args.mlm_probability)
    data_loader = torch.utils.data.DataLoader(
        tokenized, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=data_collator, drop_last=True
    )

    config = BertConfig.from_pretrained(args.model_name_or_path)
    model = DualMaskedBertForPreTraining(config)
    model.bert.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    num_update_steps_per_epoch = len(data_loader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * max_train_steps), num_training_steps=max_train_steps
    )

    model, optimizer, data_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, data_loader, lr_scheduler
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    model.train()
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(data_loader):
            with torch.no_grad():
                # 元入力から hidden を取得
                original_outputs = model.bert(
                    input_ids=batch["original_input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True,
                )
                hidden_labels = original_outputs.last_hidden_state  # (B,L,H)

            # マスク位置以外は 0 に
            mask_positions = batch["labels_token"] != -100  # (B,L)
            labels_vector = torch.zeros_like(hidden_labels)
            labels_vector[mask_positions] = hidden_labels[mask_positions]

            loss_dict = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels_token=batch["labels_token"],
                labels_vector=labels_vector,
            )
            loss = loss_dict["loss"]
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})

        # epoch 終了時に保存
        if accelerator.is_main_process:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(output_dir / f"checkpoint-epoch{epoch+1}")
            tokenizer.save_pretrained(output_dir / f"checkpoint-epoch{epoch+1}")

    # 最終保存
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        accelerator.unwrap_model(model).save_pretrained(output_dir / "final")
        tokenizer.save_pretrained(output_dir / "final")

    # ---- optional auto shutdown ----
    if args.auto_shutdown and accelerator.is_main_process:
        print("Training complete. Shutting down machine in 1 minute...")
        # 実行ユーザに電源オフ権限があれば停止できる。
        # バックグラウンドで sleep を挟んでから poweroff
        import subprocess, shlex
        cmd = "sleep 60 && poweroff"
        subprocess.Popen(cmd, shell=True)


if __name__ == "__main__":
    main()
