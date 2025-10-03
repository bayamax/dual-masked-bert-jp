from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import get_device, get_dtype


@dataclass
class HFObjects:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device
    dtype: torch.dtype


def _select_torch_dtype(requested: str, device: torch.device) -> torch.dtype:
    req = get_dtype(requested)
    if device.type == "cpu" and req in (torch.float16, torch.bfloat16):
        return torch.float32
    return req


def load_tokenizer_and_model(
    model_name: str,
    requested_dtype: str = "float16",
    device: torch.device | None = None,
    attn_implementation: str = "eager",
) -> HFObjects:
    device = device or get_device()
    dtype = _select_torch_dtype(requested_dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        # LLaMA系はpadが未設定なことが多いためEOSを使用
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        )
    except TypeError:
        # 古いtransformersでは引数未対応
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    model.eval()

    if device.type != "cpu":
        model.to(device)

    # generateで必須になることがある
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 可能であればconfig側にも反映
    try:
        model.config.attn_implementation = attn_implementation
    except Exception:
        pass
    return HFObjects(tokenizer=tokenizer, model=model, device=device, dtype=dtype)


def get_bos_prefixed_input_ids(tokenizer: AutoTokenizer, token_ids: torch.Tensor, add_bos: bool) -> torch.Tensor:
    """tokenごとに [BOS?, token] からなる2列または1列のバッチ入力を作成。
    token_ids: [B]
    return: [B, L]
    """
    if add_bos and tokenizer.bos_token_id is not None:
        bos = torch.full_like(token_ids, tokenizer.bos_token_id)
        return torch.stack([bos, token_ids], dim=1)
    return token_ids.unsqueeze(1)


