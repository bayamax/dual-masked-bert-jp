from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .hf_model import load_tokenizer_and_model


@torch.inference_mode()
def generate_with_estar(
    hf,
    e_star: torch.Tensor,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    prefix_len_for_alignment: int | None = None,
) -> str:
    tok = hf.tokenizer
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(hf.device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(hf.device)

    layer = hf.model.get_input_embeddings()
    inp_emb = layer(input_ids)  # [1, L, d]

    # e* は [k, d] を想定
    e_star = e_star.to(device=inp_emb.device, dtype=inp_emb.dtype)
    virt = e_star.unsqueeze(0)  # [1, k, d]
    full_emb = torch.cat([virt, inp_emb], dim=1)

    if attn_mask is None:
        attn_full = torch.ones(full_emb.size()[:2], dtype=torch.long, device=full_emb.device)
    else:
        virt_mask = torch.ones((attn_mask.size(0), e_star.size(0)), dtype=attn_mask.dtype, device=attn_mask.device)
        attn_full = torch.cat([virt_mask, attn_mask], dim=1)

    gen_kwargs = {}
    # 位置合わせ（学習時のprefix末尾に仮想トークンを置いた絶対位置へ合わせる）
    if prefix_len_for_alignment is not None:
        P = int(prefix_len_for_alignment)
        k = int(e_star.size(0))
        L = int(inp_emb.size(1))
        start = max(0, P - k)
        virt_pos = torch.arange(start, start + k, dtype=torch.long, device=full_emb.device)
        foll_pos = torch.arange(P, P + L, dtype=torch.long, device=full_emb.device)
        pos_ids = torch.cat([virt_pos, foll_pos], dim=0).unsqueeze(0)
        gen_kwargs["position_ids"] = pos_ids

    try:
        out_ids = hf.model.generate(
            inputs_embeds=full_emb,
            attention_mask=attn_full,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            **gen_kwargs,
        )
    except RuntimeError:
        # 位置合わせによる内部マスク長不整合が起きた場合は相対位置にフォールバック
        out_ids = hf.model.generate(
            inputs_embeds=full_emb,
            attention_mask=attn_full,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out_ids[0], skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--e_star_path", type=Path, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    args = p.parse_args()

    hf = load_tokenizer_and_model(args.model_name)
    obj = torch.load(args.e_star_path, map_location="cpu")
    e_star = obj["e_star"]  # [k, d]
    meta = obj.get("meta", {})
    P = meta.get("prefix_len")

    text = generate_with_estar(
        hf,
        e_star,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prefix_len_for_alignment=(int(P) if P is not None else None),
    )
    print(text)


if __name__ == "__main__":
    main()


