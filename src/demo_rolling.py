from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import torch

from .hf_model import load_tokenizer_and_model


def load_converter(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt["meta"]
    from .train_converter import build_model

    model = build_model(meta["model_type"], meta["d_model"], meta.get("hidden_dim", 2048))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


def compute_avg_input_norm(hf) -> float:
    emb = hf.model.get_input_embeddings().weight.detach().to(torch.float32)
    return emb.norm(dim=-1).mean().item()


@torch.inference_mode()
def compress_text_to_virtual_token(hf, text: str, converter, target_norm: float | None = None, pool: str = "last") -> torch.Tensor:
    tok = hf.tokenizer
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(hf.device)
    attn_mask = enc["attention_mask"].to(hf.device)

    out = hf.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    h = out.hidden_states[-1]
    if pool == "last":
        lengths = attn_mask.sum(dim=1) - 1
        idx = lengths.clamp(min=0).item()
        rep = h[0, idx, :].unsqueeze(0)
    else:
        mask = attn_mask.bool()
        rep = h[mask].mean(dim=0, keepdim=True)

    conv_param = next(converter.parameters())
    virt = converter(rep.to(device=conv_param.device, dtype=conv_param.dtype))  # [1, d]

    # ノルム整合（任意）
    if target_norm is not None:
        v = virt.norm(dim=-1, keepdim=True) + 1e-8
        virt = virt / v * target_norm
    # モデルの入力埋め込みdtype/deviceへ
    input_emb_layer = hf.model.get_input_embeddings()
    emb_weight = input_emb_layer.weight
    virt = virt.to(device=emb_weight.device, dtype=emb_weight.dtype)
    return virt  # [1, d]


def split_sentences(text: str) -> Tuple[List[str], str]:
    # シンプルな日本語/英語混在の文分割（句点や?!、改行）
    # 完璧ではないがデモ用途には十分
    pattern = r"([^。！？!?\n]+[。！？!?\n])"
    parts = re.findall(pattern, text)
    consumed = "".join(parts)
    rest = text[len(consumed):]
    return parts, rest


@torch.inference_mode()
def build_prefix_inputs_embeds(hf, virtual_embeds: List[torch.Tensor], residual_text: str) -> torch.Tensor:
    tok = hf.tokenizer
    enc = tok(residual_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(hf.device)
    input_embed_layer = hf.model.get_input_embeddings()
    inp_emb = input_embed_layer(input_ids)  # [1, L, d]

    if len(virtual_embeds) == 0:
        return inp_emb
    virt_stack = torch.cat([v for v in virtual_embeds], dim=0).unsqueeze(0)  # [1, V, d]
    virt_stack = virt_stack.to(device=inp_emb.device, dtype=inp_emb.dtype)
    full = torch.cat([virt_stack, inp_emb], dim=1)  # [1, V+L, d]
    return full


@torch.inference_mode()
def generate_rolling(
    converter_path: Path,
    model_name: str,
    prompt: str,
    max_new_tokens: int = 256,
    window_tokens: int = 512,
    step_tokens: int = 32,
    virtual_per_sentence: int = 1,
    norm_align: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    hf = load_tokenizer_and_model(model_name)
    converter, _ = load_converter(converter_path)
    converter.to(hf.device)
    converter.eval()

    target_norm = compute_avg_input_norm(hf) if norm_align else None

    # 未圧縮の文と未完フラグメント
    uncompressed_sentences: List[str] = []
    fragment = ""
    init_parts, init_rest = split_sentences(prompt)
    uncompressed_sentences.extend(init_parts)
    fragment += init_rest

    compressed_virt: List[torch.Tensor] = []  # 各文1ベクトル（[1, d]）

    # 進捗テキスト
    generated_accum = ""

    tok = hf.tokenizer
    while len(tok.encode(generated_accum)) < max_new_tokens:
        residual_text = "".join(uncompressed_sentences) + fragment + generated_accum
        # 窓制限に収めるため、古い文から圧縮
        while True:
            # トークン数を計測
            ids = tok(residual_text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(hf.device)
            total_tokens = ids.size(1) + len(compressed_virt) * virtual_per_sentence
            if total_tokens <= window_tokens or len(uncompressed_sentences) == 0:
                break
            # 最古の文を圧縮移動
            oldest = uncompressed_sentences.pop(0)
            virt = compress_text_to_virtual_token(hf, oldest, converter, target_norm=target_norm)
            compressed_virt.append(virt)
            # 再計算用にresidual_textも先頭文を削る
            residual_text = "".join(uncompressed_sentences) + fragment + generated_accum

        # プレフィックス埋め込み作成
        prefix = build_prefix_inputs_embeds(hf, compressed_virt, "".join(uncompressed_sentences) + fragment + generated_accum)

        gen = hf.model.generate(
            inputs_embeds=prefix,
            max_new_tokens=min(step_tokens, max_new_tokens - len(tok.encode(generated_accum))),
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
        )
        seq = gen.sequences[0]
        new_text = tok.decode(seq, skip_special_tokens=True)
        # 生成はprefix分のtoken_idsが無いので、ほぼ新規分のみのはずだが、
        # 念のため差分で追加
        if len(new_text) > len(prompt):
            delta = new_text[len(prompt):]
        else:
            delta = new_text
        generated_accum += delta

        # 文として確定した部分を未圧縮キューへ移動
        parts, rest = split_sentences(generated_accum)
        if parts:
            uncompressed_sentences.extend(parts)
            generated_accum = rest

        # 早期終了（EOS相当の改行/句点で止めたい場合はここで制御可能）
        if len(delta.strip()) == 0:
            break

    # 最終テキスト
    final_text = "".join(uncompressed_sentences) + fragment + generated_accum
    return final_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--converter_path", type=Path, required=True)
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--window_tokens", type=int, default=512)
    p.add_argument("--step_tokens", type=int, default=32)
    p.add_argument("--virtual_per_sentence", type=int, default=1)
    p.add_argument("--no_norm_align", dest="norm_align", action="store_false")
    p.add_argument("--norm_align", action="store_true", default=True)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    args = p.parse_args()

    out = generate_rolling(
        converter_path=args.converter_path,
        model_name=args.model_name,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        window_tokens=args.window_tokens,
        step_tokens=args.step_tokens,
        virtual_per_sentence=args.virtual_per_sentence,
        norm_align=args.norm_align,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(out)


if __name__ == "__main__":
    main()


