from typing import Optional, Dict, List, Any
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
import random


@dataclass
class DualMaskDataCollator:
    """トークンマスクと対応する vector マスク情報を生成する"""

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15

    def torch_mask_tokens(self, inputs: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """huggingface DataCollatorForLanguageModeling と同等のマスク適用"""
        labels = inputs.clone()

        # マスク候補を選択 (special token は除外)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 非マスク箇所は -100

        # 80% [MASK], 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        # 残り 10% は元のまま
        return inputs, labels

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # examples: list of dict with 'input_ids'
        batch_inputs = [torch.tensor(e["input_ids"], dtype=torch.long) for e in examples]
        batch = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = batch.ne(self.tokenizer.pad_token_id).long()

        masked_inputs, labels_token = self.torch_mask_tokens(batch.clone())
        return {
            "input_ids": masked_inputs,
            "attention_mask": attention_mask,
            "labels_token": labels_token,
            "original_input_ids": batch,  # 元文を later use for vector label取得
        }
