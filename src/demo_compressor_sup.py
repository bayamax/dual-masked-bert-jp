from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .hf_model import load_tokenizer_and_model
from .hyperprompt import HyperPromptNet


class QueryAttentionCompressor(torch.nn.Module):
    def __init__(self, d_model: int, num_virtual: int = 1, num_heads: int = 8):
        super().__init__()
        self.num_virtual = num_virtual
        self.query = torch.nn.Parameter(torch.randn(num_virtual, d_model))
        torch.nn.init.xavier_uniform_(self.query)
        self.attn = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.out = torch.nn.Sequential(torch.nn.Linear(d_model, d_model), torch.nn.LayerNorm(d_model))

    def forward(self, ctx_states: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, l, d = ctx_states.size()
        q = self.query.unsqueeze(0).expand(b, -1, -1).to(dtype=ctx_states.dtype, device=ctx_states.device)
        # Inference-safe tensors for MHA (avoid saving inference tensors for backward)
        q = q.contiguous().clone()
        k = ctx_states.contiguous().clone()
        kpm = None
        if attn_mask is not None:
            kpm = (attn_mask == 0)
        attn_out, _ = self.attn(query=q, key=k, value=k, key_padding_mask=kpm, need_weights=False)
        return self.out(attn_out)  # [B, k, d]


@torch.inference_mode()
def build_ctx_hidden(hf, text: str, ctx_len: int, hidden_layer_index: int = -2) -> tuple[torch.Tensor, torch.Tensor]:
    tok = hf.tokenizer
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(hf.device)
    attn_mask = enc.get("attention_mask").to(hf.device)
    out = hf.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    h = out.hidden_states[hidden_layer_index][0]  # [L, d]
    L, d = h.size(0), h.size(1)
    H = torch.zeros((ctx_len, d), dtype=h.dtype, device=hf.device)
    M = torch.zeros((ctx_len,), dtype=torch.long, device=hf.device)
    if L >= ctx_len:
        H[:] = h[L - ctx_len : L]
        M[:] = 1
    else:
        H[-L:] = h
        M[-L:] = 1
    return H.unsqueeze(0), M.unsqueeze(0)  # [1, L, d], [1, L]


@torch.inference_mode()
def generate_with_estar(
    hf,
    e_star: torch.Tensor,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    prefix_len_for_alignment: int | None = None,
    repetition_penalty: float = 1.1,
    do_sample: bool = False,
) -> str:
    tok = hf.tokenizer
    # Chatテンプレート対応（Chat系モデルではこちらを優先）
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": prompt}]
        input_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(hf.device)
        attn_mask = None
    else:
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(hf.device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(hf.device)

    layer = hf.model.get_input_embeddings()
    inp_emb = layer(input_ids)  # [1, L, d]

    e_star = e_star.to(device=inp_emb.device, dtype=inp_emb.dtype)  # [1, k, d]
    full_emb = torch.cat([e_star, inp_emb], dim=1)

    if attn_mask is None:
        attn_full = torch.ones(full_emb.size()[:2], dtype=torch.long, device=full_emb.device)
    else:
        virt_mask = torch.ones((attn_mask.size(0), e_star.size(1)), dtype=attn_mask.dtype, device=attn_mask.device)
        attn_full = torch.cat([virt_mask, attn_mask], dim=1)

    gen_kwargs = {}
    if prefix_len_for_alignment is not None:
        P = int(prefix_len_for_alignment)
        k = int(e_star.size(1))
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
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            **gen_kwargs,
        )
    except Exception:
        # 位置IDが不整合になる場合は相対位置（position_idsなし）にフォールバック
        out_ids = hf.model.generate(
            inputs_embeds=full_emb,
            attention_mask=attn_full,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out_ids[0], skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--compressor_path", type=Path, required=True)
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--ctx_len", type=int, default=64)
    p.add_argument("--hidden_layer_index", type=int, default=-2)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "auto"])
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    # アーキテクチャ選択
    p.add_argument("--arch", type=str, default="hyper", choices=["hyper", "qattn"])
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--hyper_num_layers", type=int, default=2)
    p.add_argument("--hyper_ffn_dim", type=int, default=2048)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", default=False)
    args = p.parse_args()

    # デバイス/精度を明示して整合
    import torch
    dev = None if args.device == "auto" else torch.device(args.device)
    hf = load_tokenizer_and_model(args.model_name, requested_dtype=args.dtype, device=dev)

    ckpt = torch.load(args.compressor_path, map_location="cpu")
    d_model = int(ckpt["meta"]["d_model"]) if "meta" in ckpt and "d_model" in ckpt["meta"] else hf.model.get_input_embeddings().weight.shape[1]
    num_virtual = int(ckpt["meta"].get("num_virtual", 1)) if "meta" in ckpt else 1
    meta_arch = (ckpt.get("meta", {}).get("arch", args.arch) if isinstance(ckpt.get("meta", {}), dict) else args.arch).lower()
    if meta_arch == "hyper":
        model = HyperPromptNet(
            d_model=d_model,
            num_virtual=num_virtual,
            num_heads=int(ckpt.get("meta", {}).get("num_heads", args.num_heads)),
            num_layers=int(ckpt.get("meta", {}).get("hyper_num_layers", args.hyper_num_layers)),
            ffn_dim=int(ckpt.get("meta", {}).get("hyper_ffn_dim", args.hyper_ffn_dim)),
            dropout=float(ckpt.get("meta", {}).get("dropout", 0.1)),
            prenorm=bool(ckpt.get("meta", {}).get("hyper_prenorm", True)),
        )
    else:
        model = QueryAttentionCompressor(d_model=d_model, num_virtual=num_virtual, num_heads=args.num_heads)
    model.load_state_dict(ckpt["state_dict"])
    # 圧縮器のdtypeもLLMのdtypeに合わせる
    model.to(device=hf.device, dtype=hf.dtype)
    model.eval()

    H, M = build_ctx_hidden(hf, args.text, ctx_len=args.ctx_len, hidden_layer_index=args.hidden_layer_index)
    e_star = model(H, M)  # [1, 1, d]
    # [1, k, d]へ一般化
    k = int(e_star.size(1))
    # ノルム整合：入力埋め込みの平均ノルムにe*を合わせる（学習時と同様）
    emb_layer = hf.model.get_input_embeddings()
    with torch.no_grad():
        avg_norm = emb_layer.weight.detach().to(torch.float32).norm(dim=-1).mean().to(hf.device)
    # αスケール（強度調整）
    e_star = e_star / (e_star.norm(dim=-1, keepdim=True) + 1e-8) * (avg_norm * float(args.alpha)).to(dtype=e_star.dtype)
    # prefix length for alignment
    # Chatテンプレートに合わせてprefix長を算出
    if getattr(hf.tokenizer, "chat_template", None):
        pref_ids = hf.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.text}],
            add_generation_prompt=False,
            return_tensors="pt",
        )
        pref_len = int(pref_ids.size(1))
    else:
        pref_enc = hf.tokenizer(args.text, return_tensors="pt", add_special_tokens=True)
        pref_len = int(pref_enc["input_ids"].size(1))
    text = generate_with_estar(
        hf,
        e_star,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prefix_len_for_alignment=pref_len,
        repetition_penalty=args.repetition_penalty,
        do_sample=(not args.greedy),
    )
    print(text)


if __name__ == "__main__":
    main()


