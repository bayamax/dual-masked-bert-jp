from __future__ import annotations

import argparse
from pathlib import Path
import glob
import random
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .hf_model import load_tokenizer_and_model
from .optimize_virtual_token import build_position_ids_for_virtual, optimize_virtual
from .build_e_star_dataset import load_pairs_from_jsonl, default_pairs
from .merge_corpus import load_agg as _mc_load_agg, load_stream as _mc_load_stream, dedup_samples as _mc_dedup
from .hyperprompt import HyperPromptNet


class QueryAttentionCompressor(nn.Module):
    def __init__(self, d_model: int, num_virtual: int = 1, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_virtual = num_virtual
        self.query = nn.Parameter(torch.randn(num_virtual, d_model))
        nn.init.xavier_uniform_(self.query)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, ctx_states: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # ctx_states: [B, L, d]; attn_mask: [B, L] (1=valid,0=pad) or None
        B, L, D = ctx_states.size()
        q = self.query.unsqueeze(0).expand(B, -1, -1).to(dtype=ctx_states.dtype, device=ctx_states.device)
        # convert mask to key_padding_mask (True = pad)
        kpm = None
        if attn_mask is not None:
            kpm = (attn_mask == 0)
        attn_out, _ = self.attn(query=q, key=ctx_states, value=ctx_states, key_padding_mask=kpm, need_weights=False)
        return self.out(attn_out)  # [B, k, d]


def cosine_norm_loss(pred: torch.Tensor, targ: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    # 安定のためfloat32で計算
    pred = pred.float()
    targ = targ.float()
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(targ, dim=-1)
    cos = (pred_n * targ_n).sum(dim=-1)
    norm_l2 = (pred.norm(dim=-1) - targ.norm(dim=-1)).pow(2)
    return (1.0 - cos + alpha * norm_l2).mean()


def load_dataset(path: Path):
    obj = torch.load(path)
    samples = obj["samples"]
    H = torch.stack([s["Hctx"].to(torch.float32) for s in samples], dim=0)  # [N, L, d]
    M = torch.stack([s["Hmask"].to(torch.long) for s in samples], dim=0)     # [N, L]
    # e_starは[k, d]として保存されている想定（k>=1）。そのまま積んで [N, k, d] に揃える。
    E = torch.stack([s["e_star"].to(torch.float32) for s in samples], dim=0)  # [N, k, d]
    texts: List[str] = [s.get("text", "") for s in samples]
    follows: List[str] = [s.get("follow_text", "") for s in samples]
    return H, M, E, texts, follows


def _list_stream_sample_paths(stream_dir: Path) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(stream_dir / "sample_*.pt")))]


def _load_samples_from_paths(paths: list[Path], start: int, end: int):
    H_list: list[torch.Tensor] = []
    M_list: list[torch.Tensor] = []
    E_list: list[torch.Tensor] = []
    T_list: list[str] = []
    F_list: list[str] = []
    for p in paths[start:end]:
        s = torch.load(p, map_location="cpu")
        H_list.append(s["Hctx"].to(torch.float32))
        M_list.append(s["Hmask"].to(torch.long))
        E_list.append(s["e_star"].to(torch.float32))
        T_list.append(s.get("text", ""))
        F_list.append(s.get("follow_text", ""))
    if not H_list:
        return None
    H = torch.stack(H_list, dim=0)
    M = torch.stack(M_list, dim=0)
    E = torch.stack(E_list, dim=0)
    return H, M, E, T_list, F_list


def _auto_merge_to_agg(auto_out: Path, base_agg: Path | None, stream_dir: Path) -> None:
    samples = []
    try:
        if base_agg is not None and isinstance(base_agg, Path) and base_agg.exists():
            samples.extend(_mc_load_agg(base_agg))
    except Exception:
        pass
    try:
        if stream_dir is not None and isinstance(stream_dir, Path) and stream_dir.exists():
            samples.extend(_mc_load_stream(stream_dir))
    except Exception:
        pass
    if not samples:
        return
    uniq = _mc_dedup(samples)
    auto_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": uniq}, auto_out)
    print({"auto_merge_saved": str(auto_out), "num": len(uniq)}, flush=True)


class SupDataset(Dataset):
    def __init__(self, H: torch.Tensor, M: torch.Tensor, E: torch.Tensor, texts: List[str], follows: List[str]):
        self.H = H
        self.M = M
        self.E = E
        self.texts = texts
        self.follows = follows

    def __len__(self) -> int:
        return self.H.size(0)

    def __getitem__(self, idx: int):
        return self.H[idx], self.M[idx], self.E[idx], self.texts[idx], self.follows[idx]


def _get_llama_layers(model: nn.Module) -> List[nn.Module]:
    # Try common paths: LlamaForCausalLM.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        return list(model.base_model.model.layers)
    raise RuntimeError("Unsupported model structure: cannot locate layers")


def _capture_attn_out(model: nn.Module, layers_idx: List[int], detach: bool) -> tuple[list, dict[int, torch.Tensor]]:
    layers = _get_llama_layers(model)
    hooks = []
    bufs: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def _hook(mod, inp, out):
            out0 = out[0] if isinstance(out, tuple) else out
            out0 = out0.detach() if detach else out0
            bufs[li] = out0
            return out
        return _hook

    for li in layers_idx:
        li_n = li if li >= 0 else len(layers) + li
        attn = layers[li_n].self_attn
        hooks.append(attn.register_forward_hook(make_hook(li_n)))
    return hooks, bufs


def _capture_attn_weights(model: nn.Module, layers_idx: List[int], detach: bool) -> tuple[list, dict[int, torch.Tensor]]:
    layers = _get_llama_layers(model)
    hooks = []
    bufs: dict[int, torch.Tensor] = {}

    def make_hook(li: int):
        def _hook(mod, inp, out):
            attn_w = None
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attn_w = out[1]
            if attn_w is not None:
                attn_w = attn_w.detach() if detach else attn_w
                bufs[li] = attn_w
            return out
        return _hook

    for li in layers_idx:
        li_n = li if li >= 0 else len(layers) + li
        attn = layers[li_n].self_attn
        hooks.append(attn.register_forward_hook(make_hook(li_n)))
    return hooks, bufs


def _safe_kl(sl_t: torch.Tensor, sl_s: torch.Tensor, temperature: float) -> torch.Tensor:
    # float32で安定計算
    logprobs_t = F.log_softmax(sl_t.float() / temperature, dim=-1)
    probs_t = torch.softmax(sl_t.float() / temperature, dim=-1)
    logprobs_s = F.log_softmax(sl_s.float() / temperature, dim=-1)
    kl = (-(probs_t * logprobs_s).sum(dim=-1)).mean()
    return kl


def train(dataset_path: Path, save_path: Path, epochs: int = 200, batch_size: int = 8, lr: float = 1e-4, kl_weight: float = 0.2, num_virtual: int = 1, device_str: str = "cpu", requested_dtype: str = "float32", log_every: int = 1, dropout: float = 0.1, kl_temperature: float = 1.0, hidden_weight: float = 0.0, hidden_layer_index: int = -1, hidden_loss_type: str = "cosine", attn_weight: float = 0.0, attn_layers: str = "", attn_use_weights: bool = False, resume_path: Path | None = None, arch: str = "hyper", num_heads: int = 8, hyper_num_layers: int = 2, hyper_ffn_dim: int = 2048, hyper_prenorm: bool = True, vec_weight: float = 1.0, norm_penalty: float = 0.02, data_growth_per_epoch: int = 0, data_growth_start: int = 0, data_source: str = "file", stream_dir: Path | None = None, gen_per_epoch: int = 0, gen_steps: int = 60, gen_pairs_jsonl: Path | None = None, gen_init_mode: str = "avg_embed", gen_ctx_len: int = 64, gen_hidden_weight: float = 0.0, gen_hidden_layer_index: int = -2, gen_hidden_loss_type: str = "cosine", gen_pairs_mode: str = "default", gen_source_stream_dir: Path | None = None, gen_unique: bool = True) -> None:
    # device / hf
    dev = torch.device(device_str)
    hf = load_tokenizer_and_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        requested_dtype=requested_dtype,
        device=dev,
        attn_implementation="eager",
    )
    device = hf.device
    hf.model.eval()
    hf.model.requires_grad_(False)
    tok = hf.tokenizer

    # load data
    data_source = (data_source or "file").lower()
    if data_source == "stream":
        if stream_dir is None:
            raise ValueError("stream_dir is required when data_source=stream")
        stream_paths = _list_stream_sample_paths(stream_dir)
        cached_HME: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[str], list[str]] | None = ([], [], [], [], [])
        cached_head = 0
        # peek first sample to get dims
        if len(stream_paths) == 0:
            raise RuntimeError(f"no samples found in {stream_dir}")
        peek = torch.load(stream_paths[0], map_location="cpu")
        L = int(peek["Hctx"].size(0))
        d = int(peek["Hctx"].size(1))
        N_total = len(stream_paths)
    elif data_source == "generate":
        # set dims from current model embeddings
        d = int(hf.model.get_input_embeddings().weight.shape[1])
        L = int(gen_ctx_len)
        # pairs source
        pairs_mode = (gen_pairs_mode or "default").lower()
        pairs: list[dict] = []
        if pairs_mode == "jsonl" and (gen_pairs_jsonl is not None) and Path(gen_pairs_jsonl).exists():
            pairs = load_pairs_from_jsonl(gen_pairs_jsonl)
        elif pairs_mode == "dataset" and isinstance(dataset_path, Path) and str(dataset_path) != "" and dataset_path.exists():
            try:
                Hds, Mds, Eds, Tds, Fds = load_dataset(dataset_path)
                pairs = [{"text": str(t), "follow_text": str(f)} for t, f in zip(Tds, Fds)]
            except Exception:
                pairs = default_pairs()
        elif pairs_mode == "stream" and (gen_source_stream_dir is not None) and Path(gen_source_stream_dir).exists():
            try:
                src_paths = _list_stream_sample_paths(gen_source_stream_dir)
                for pth in src_paths:
                    s = torch.load(pth, map_location="cpu")
                    pairs.append({"text": str(s.get("text", "")), "follow_text": str(s.get("follow_text", ""))})
            except Exception:
                pairs = default_pairs()
        else:
            pairs = default_pairs()
        # shuffle pairs once per run
        random.shuffle(pairs)
        pair_cursor = 0
        used_pairs: set[tuple[str, str]] = set()
        # generated cache
        gen_H_list: list[torch.Tensor] = []
        gen_M_list: list[torch.Tensor] = []
        gen_E_list: list[torch.Tensor] = []
        gen_T_list: list[str] = []
        gen_F_list: list[str] = []
        # base dataset (optional): 既存コーパスを常に含める
        base_H_stack: torch.Tensor | None = None
        base_M_stack: torch.Tensor | None = None
        base_E_stack: torch.Tensor | None = None
        base_T_list: list[str] = []
        base_F_list: list[str] = []
        if isinstance(dataset_path, Path) and str(dataset_path) != "" and dataset_path.exists():
            H0, M0, E0, T0, F0 = load_dataset(dataset_path)
            base_H_stack = H0
            base_M_stack = M0
            base_E_stack = E0
            base_T_list = list(T0)
            base_F_list = list(F0)
        # optional stream save seed (append-only)
        gen_stream_count = 0
        if stream_dir is not None:
            stream_dir.mkdir(parents=True, exist_ok=True)
            gen_stream_count = len(_list_stream_sample_paths(stream_dir))
    else:
        H, M, E, texts, follows = load_dataset(dataset_path)
        N, L, d = H.size()
    # model for compressor / KL uses hf above
    # 圧縮器の選択（ハイパーネット or シンプルQAttn）
    arch = (arch or "hyper").lower()
    if arch == "hyper":
        model = HyperPromptNet(
            d_model=d,
            num_virtual=num_virtual,
            num_heads=num_heads,
            num_layers=hyper_num_layers,
            ffn_dim=hyper_ffn_dim,
            dropout=dropout,
            prenorm=hyper_prenorm,
        )
    else:
        model = QueryAttentionCompressor(d_model=d, num_virtual=num_virtual, num_heads=num_heads, dropout=dropout)
    model.to(device=hf.device, dtype=hf.dtype)
    # resume (partial, shape-safe)
    if resume_path is not None and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        own = model.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        loaded: list[str] = []
        for k, v in state.items():
            if (k in own) and (tuple(v.shape) == tuple(own[k].shape)):
                filtered[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        missing = [k for k in own.keys() if k not in filtered]
        model.load_state_dict(filtered, strict=False)
        print({
            "status": "resumed",
            "path": str(resume_path),
            "loaded": int(len(loaded)),
            "skipped": int(len(skipped)),
            "missing": int(len(missing)),
        }, flush=True)
    tok = hf.tokenizer
    emb_layer = hf.model.get_input_embeddings()
    # 予測e*のノルムを整えるため入力埋め込みの平均ノルムを事前計算
    with torch.no_grad():
        avg_norm = emb_layer.weight.detach().to(torch.float32).norm(dim=-1).mean().to(device)

    def collate(batch):
        Hs, Ms, Es, Ts, Fs = zip(*batch)
        return torch.stack(Hs, 0), torch.stack(Ms, 0), torch.stack(Es, 0), list(Ts), list(Fs)

    # dataset/dataloader will be prepared per-epoch if growth/stream/generate
    if (data_growth_per_epoch <= 0) and (data_source == "file"):
        ds = SupDataset(H, M, E, texts, follows)
        n_val = max(1, int(0.2 * ds.__len__()))
        n_train = ds.__len__() - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=min(batch_size, n_train), shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=n_val, shuffle=False, collate_fn=collate)
    else:
        train_loader = None
        val_loader = None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    best = float("inf")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    attn_layers_idx: List[int] = []
    if attn_layers:
        try:
            attn_layers_idx = [int(x.strip()) for x in attn_layers.split(",") if x.strip()]
        except Exception:
            attn_layers_idx = []

    def attn_loss_from_bufs(buf_t: dict[int, torch.Tensor], buf_s: dict[int, torch.Tensor], P: int, Lf: int, k: int) -> torch.Tensor:
        if not buf_t or not buf_s or not attn_layers_idx or attn_weight <= 0.0:
            return torch.tensor(0.0, device=device, dtype=hf.dtype)
        losses = []
        for li in attn_layers_idx:
            li_n = li if li >= 0 else len(_get_llama_layers(hf.model)) + li
            if (li_n not in buf_t) or (li_n not in buf_s):
                continue
            at_t = buf_t[li_n].float()
            at_s = buf_s[li_n].float()
            # teacherは勾配不要
            at_t = at_t.detach()
            # フォロートークン自身のクエリ位置で比較（Q/K/V再現により直接的）
            sl_t = at_t[:, P : P + Lf, :]
            sl_s = at_s[:, k : k + Lf, :]
            losses.append(1.0 - F.cosine_similarity(F.normalize(sl_s, dim=-1), F.normalize(sl_t, dim=-1), dim=-1).mean())
        if not losses:
            return torch.tensor(0.0, device=device, dtype=hf.dtype)
        return torch.stack(losses).mean()

    def attn_weights_loss_from_bufs(buf_t: dict[int, torch.Tensor], buf_s: dict[int, torch.Tensor], P: int, Lf: int, k: int) -> torch.Tensor:
        # attn_weights: [B, H, Q, K] を想定（transformers実装依存）
        if not buf_t or not buf_s or not attn_layers_idx or attn_weight <= 0.0:
            return torch.tensor(0.0, device=device, dtype=hf.dtype)
        losses = []
        for li in attn_layers_idx:
            li_n = li if li >= 0 else len(_get_llama_layers(hf.model)) + li
            if (li_n not in buf_t) or (li_n not in buf_s):
                continue
            wt_t = buf_t[li_n].float().detach()  # [B, H, Q, K]
            wt_s = buf_s[li_n].float()
            # 教師: フォロートークンのクエリ範囲 [P .. P+Lf-1] が prefixへどれだけ注意しているか（K: [0 .. P-1]）
            q_t = wt_t[:, :, P : P + Lf, :]  # [B,H,Lf,Ktot]
            q_s = wt_s[:, :, k : k + Lf, :]
            if q_t.size(-1) < P or q_s.size(-1) < k:
                continue
            mass_t = q_t[..., :P].sum(dim=-1)  # [B,H,Lf]
            mass_s = q_s[..., :k].sum(dim=-1)
            # ヘッド平均でMSE（直接的に「prefixへ寄る度合い」を合わせる）
            losses.append(F.mse_loss(mass_s.mean(dim=1), mass_t.mean(dim=1)))
        if not losses:
            return torch.tensor(0.0, device=device, dtype=hf.dtype)
        return torch.stack(losses).mean()

    def hidden_match_loss(out_t, out_s, P: int, Lf: int, k: int) -> torch.Tensor:
        if hidden_weight <= 0.0:
            return torch.tensor(0.0, device=device, dtype=hf.dtype)
        try:
            h_t_full = out_t.hidden_states[hidden_layer_index]
            h_s_full = out_s.hidden_states[hidden_layer_index]
        except Exception:
            return torch.tensor(0.0, device=device, dtype=hf.dtype)
        # 直前位置スライスで整合
        h_t = h_t_full[:, P - 1 : P - 1 + Lf, :].float()
        h_s = h_s_full[:, k - 1 : k - 1 + Lf, :].float()
        if hidden_loss_type == "mse":
            return F.mse_loss(h_s, h_t)
        elif hidden_loss_type == "cosine":
            pred_n = F.normalize(h_s, dim=-1)
            targ_n = F.normalize(h_t, dim=-1)
            return (1.0 - (pred_n * targ_n).sum(dim=-1)).mean()
        elif hidden_loss_type == "cosine_norm":
            pred_n = F.normalize(h_s, dim=-1)
            targ_n = F.normalize(h_t, dim=-1)
            cos = (pred_n * targ_n).sum(dim=-1)
            norm_l2 = (h_s.norm(dim=-1) - h_t.norm(dim=-1)).pow(2)
            return (1.0 - cos + 0.1 * norm_l2).mean()
        else:
            return torch.tensor(0.0, device=device, dtype=hf.dtype)

    for ep in range(1, epochs + 1):
        # Prepare dataset per epoch if growth/stream
        if (data_growth_per_epoch > 0) or (data_source in ("stream", "generate")):
            if data_source == "stream":
                # compute head
                head = data_growth_start + (ep - 1) * max(0, data_growth_per_epoch)
                head = max(1, min(head if head > 0 else len(stream_paths), len(stream_paths)))
                # load new samples incrementally
                if cached_HME is not None:
                    H_list, M_list, E_list, T_list, F_list = cached_HME
                else:
                    H_list, M_list, E_list, T_list, F_list = [], [], [], [], []
                if head > cached_head:
                    delta = _load_samples_from_paths(stream_paths, cached_head, head)
                    if delta is not None:
                        Hd, Md, Ed, Td, Fd = delta
                        H_list.extend([t for t in Hd])
                        M_list.extend([t for t in Md])
                        E_list.extend([t for t in Ed])
                        T_list.extend(Td)
                        F_list.extend(Fd)
                        cached_head = head
                        cached_HME = (H_list, M_list, E_list, T_list, F_list)
                # stack
                H_ep = torch.stack(H_list, dim=0) if H_list else None
                M_ep = torch.stack(M_list, dim=0) if M_list else None
                E_ep = torch.stack(E_list, dim=0) if E_list else None
                texts_ep = list(T_list)
                follows_ep = list(F_list)
            elif data_source == "generate":
                # generate new samples this epoch
                def _build_ctx_hidden_for_text(hf_local, text: str, ctx_len: int, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
                    tok = hf_local.tokenizer
                    enc = tok(text, return_tensors="pt", add_special_tokens=True)
                    input_ids = enc["input_ids"].to(hf_local.device)
                    attn_mask = enc.get("attention_mask").to(hf_local.device)
                    out = hf_local.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
                    h_all = out.hidden_states[layer_idx][0]  # [Lx, d]
                    Lx, dx = h_all.size(0), h_all.size(1)
                    Hloc = torch.zeros((ctx_len, dx), dtype=h_all.dtype)
                    Mloc = torch.zeros((ctx_len,), dtype=torch.long)
                    if Lx >= ctx_len:
                        Hloc[:] = h_all[Lx - ctx_len : Lx].detach().cpu()
                        Mloc[:] = 1
                    else:
                        Hloc[-Lx:] = h_all.detach().cpu()
                        Mloc[-Lx:] = 1
                    return Hloc, Mloc

                n_new = int(max(0, gen_per_epoch))
                for _ in range(n_new):
                    # pick next unused pair if required
                    trials = 0
                    p = None
                    while trials < len(pairs):
                        cand = pairs[pair_cursor % len(pairs)]
                        pair_cursor += 1
                        trials += 1
                        key = (str(cand.get("text", "")), str(cand.get("follow_text", "")))
                        if (not gen_unique) or (key not in used_pairs):
                            p = cand
                            used_pairs.add(key)
                            break
                    if p is None:
                        # all exhausted; fallback to allow repeats
                        p = pairs[(pair_cursor - 1) % len(pairs)]
                    text_i = str(p.get("text", ""))
                    follow_i = str(p.get("follow_text", ""))
                    # optimize e*
                    obj = optimize_virtual(
                        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        text=text_i,
                        follow_text=follow_i,
                        num_virtual=num_virtual,
                        steps=int(gen_steps),
                        lr=0.05,
                        temperature=float(kl_temperature),
                        lambda_norm=0.05,
                        save_path=None,
                        device_override=None,
                        init_mode=str(gen_init_mode),
                        hidden_weight=float(gen_hidden_weight),
                        hidden_layer_index=int(gen_hidden_layer_index),
                        hidden_loss_type=str(gen_hidden_loss_type),
                        hf_obj=hf,
                        log_every=0,
                    )
                    e_star_i = obj["e_star"]  # [k,d]
                    Hloc, Mloc = _build_ctx_hidden_for_text(hf, text_i, int(gen_ctx_len), int(gen_hidden_layer_index))
                    gen_H_list.append(Hloc)
                    gen_M_list.append(Mloc)
                    gen_E_list.append(e_star_i.to(torch.float32))
                    gen_T_list.append(text_i)
                    gen_F_list.append(follow_i)
                    # optional stream save
                    if stream_dir is not None:
                        gen_stream_count += 1
                        out_path = stream_dir / f"sample_{gen_stream_count:06d}.pt"
                        torch.save({
                            "Hctx": Hloc,
                            "Hmask": Mloc,
                            "e_star": e_star_i.detach().to(torch.float32).cpu(),
                            "text": text_i,
                            "follow_text": follow_i,
                            "meta": obj.get("meta", {}),
                        }, out_path)
                        try:
                            mf = (stream_dir / "manifest.jsonl").open("a", encoding="utf-8")
                            mf.write("{" + f"\"idx\": {gen_stream_count}, \"path\": \"{str(out_path)}\"" + "}\n")
                            mf.close()
                        except Exception:
                            pass
                # stack all generated so far
                # stack generated so far
                if gen_H_list:
                    G_H = torch.stack(gen_H_list, dim=0)
                    G_M = torch.stack(gen_M_list, dim=0)
                    G_E = torch.stack(gen_E_list, dim=0)
                else:
                    # 生成ゼロは想定しないが、保険
                    G_H = None
                    G_M = None
                    G_E = None
                # combine base + generated
                if base_H_stack is not None:
                    if G_H is not None:
                        H_ep = torch.cat([base_H_stack, G_H], dim=0)
                        M_ep = torch.cat([base_M_stack, G_M], dim=0)
                        E_ep = torch.cat([base_E_stack, G_E], dim=0)
                        texts_ep = base_T_list + list(gen_T_list)
                        follows_ep = base_F_list + list(gen_F_list)
                    else:
                        H_ep = base_H_stack
                        M_ep = base_M_stack
                        E_ep = base_E_stack
                        texts_ep = base_T_list
                        follows_ep = base_F_list
                else:
                    if G_H is None:
                        raise RuntimeError("no generated samples")
                    H_ep = G_H
                    M_ep = G_M
                    E_ep = G_E
                    texts_ep = list(gen_T_list)
                    follows_ep = list(gen_F_list)
            else:
                # file mode with growth
                N_total = H.size(0)
                head = data_growth_start + (ep - 1) * max(0, data_growth_per_epoch)
                head = max(1, min(N_total, head if head > 0 else N_total))
                H_ep = H[:head]
                M_ep = M[:head]
                E_ep = E[:head]
                texts_ep = texts[:head]
                follows_ep = follows[:head]

            ds = SupDataset(H_ep, M_ep, E_ep, texts_ep, follows_ep)
            n_val = max(1, int(0.2 * ds.__len__()))
            n_train = ds.__len__() - n_val
            train_ds, val_ds = random_split(ds, [n_train, n_val])
            train_loader = DataLoader(train_ds, batch_size=min(batch_size, n_train), shuffle=True, collate_fn=collate)
            val_loader = DataLoader(val_ds, batch_size=n_val, shuffle=False, collate_fn=collate)
        print({"epoch": ep, "data_head": int(ds.__len__()), "n_train": int(n_train), "n_val": int(n_val)}, flush=True)
        model.train()
        total = 0.0
        for step_idx, (Hb, Mb, Eb, Tb, Fb) in enumerate(train_loader, start=1):
            Hb = Hb.to(device=hf.device, dtype=hf.dtype)
            Mb = Mb.to(device)
            Eb = Eb.to(device=hf.device, dtype=hf.dtype)
            pred = model(Hb, Mb)  # [B, k, d]
            # 未正規化e*（前向き用）と、ノルム整合版（回帰用）を分離
            pred_raw = pred
            pred_normed = pred_raw / (pred_raw.norm(dim=-1, keepdim=True) + 1e-8) * avg_norm
            # base loss: e*回帰はノルム整合版で安定化（float32で）
            # k不一致時は共通のmin(k)のみで回帰
            try:
                k_min = int(min(pred_normed.size(1), Eb.size(1)))
            except Exception:
                k_min = pred_normed.size(1)
            loss_vec = cosine_norm_loss(pred_normed[:, :k_min, :], Eb[:, :k_min, :]) if k_min > 0 else torch.tensor(0.0, device=device, dtype=hf.dtype)
            # ノルム整合ペナルティ（未正規化pred_rawのノルムを平均へ）
            loss_norm = (pred_raw.norm(dim=-1) - avg_norm).pow(2).mean()
            kl_val = None
            loss_attn = torch.tensor(0.0, device=device, dtype=hf.dtype)
            loss_hidden = torch.tensor(0.0, device=device, dtype=hf.dtype)
            # optional KL: teacher(text+follow) vs student(e*+follow)
            if kl_weight > 0.0:
                # tokenize batch of text and follow
                enc_t = tok(list(Tb), return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
                enc_f = tok(list(Fb), return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
                prefix_ids = enc_t["input_ids"].to(device)
                follow_ids = enc_f["input_ids"].to(device)
                # teacher logits slices (with optional attn hooks)
                input_ids = torch.cat([prefix_ids, follow_ids], dim=1)
                attn = torch.ones_like(input_ids)
                # 出力特徴（attn_out）と重み（attn_weights）を両方取る
                hooks_t1, bufs_t1 = _capture_attn_out(hf.model, attn_layers_idx, detach=True) if attn_layers_idx and attn_weight > 0.0 else ([], {})
                hooks_t2, bufs_t2 = (_capture_attn_weights(hf.model, attn_layers_idx, detach=True) if (attn_use_weights and attn_layers_idx and attn_weight > 0.0) else ([], {}))
                with torch.no_grad():
                    out_t = hf.model(input_ids=input_ids, attention_mask=attn, use_cache=False, output_attentions=bool(hooks_t2), output_hidden_states=(hidden_weight > 0.0))
                for h in (hooks_t1 + hooks_t2):
                    h.remove()
                logits_t = out_t.logits
                P = prefix_ids.size(1)
                Lf = follow_ids.size(1)
                sl_t = logits_t[:, P - 1 : P - 1 + Lf, :]
                # student logits slices with predicted e*
                follow_emb = emb_layer(follow_ids)
                # 前向きは正規化済みe*を使用して安定化
                e_pred = pred_normed  # [B,k,d]
                full_emb = torch.cat([e_pred, follow_emb], dim=1)
                attn_s = torch.ones(full_emb.size()[:2], dtype=torch.long, device=device)
                hooks_s1, bufs_s1 = _capture_attn_out(hf.model, attn_layers_idx, detach=False) if attn_layers_idx and attn_weight > 0.0 else ([], {})
                hooks_s2, bufs_s2 = (_capture_attn_weights(hf.model, attn_layers_idx, detach=False) if (attn_use_weights and attn_layers_idx and attn_weight > 0.0) else ([], {}))
                try:
                    k = int(e_pred.size(1))
                    P_int = int(P)
                    Lf_int = int(Lf)
                    pos_ids = build_position_ids_for_virtual(P_int, Lf_int, k).to(device)
                    if pos_ids.size(0) == 1 and full_emb.size(0) > 1:
                        pos_ids = pos_ids.expand(full_emb.size(0), -1)
                    out_s = hf.model(inputs_embeds=full_emb, attention_mask=attn_s, position_ids=pos_ids, use_cache=False, output_attentions=bool(hooks_s2), output_hidden_states=(hidden_weight > 0.0))
                except Exception:
                    out_s = hf.model(inputs_embeds=full_emb, attention_mask=attn_s, use_cache=False, output_attentions=bool(hooks_s2), output_hidden_states=(hidden_weight > 0.0))
                for h in (hooks_s1 + hooks_s2):
                    h.remove()
                logits_s = out_s.logits
                sl_s = logits_s[:, (e_pred.size(1) - 1) : (e_pred.size(1) - 1 + Lf), :]
                # KLをfloat32で安全計算
                kl = _safe_kl(sl_t, sl_s, kl_temperature)
                kl_val = kl
                # hidden一致（中間層）
                if hidden_weight > 0.0:
                    loss_hidden = hidden_match_loss(out_t, out_s, int(P), int(Lf), int(e_pred.size(1)))
                # attention output matching (optional, float32)
                if attn_layers_idx and attn_weight > 0.0:
                    loss_attn_feat = attn_loss_from_bufs(bufs_t1, bufs_s1, int(P), int(Lf), int(e_pred.size(1)))
                    if hooks_t2:
                        loss_attn_w = attn_weights_loss_from_bufs(bufs_t2, bufs_s2, int(P), int(Lf), int(e_pred.size(1)))
                        loss_attn = 0.5 * (loss_attn_feat + loss_attn_w)
                    else:
                        loss_attn = loss_attn_feat
                loss = vec_weight * loss_vec + kl_weight * kl.float() + attn_weight * loss_attn.float() + hidden_weight * loss_hidden.float() + norm_penalty * loss_norm.float()
            else:
                loss = vec_weight * loss_vec + norm_penalty * loss_norm.float()
            # NaN/Infガード
            if not torch.isfinite(loss):
                print({"warn": "skip_nan", "epoch": ep, "step": step_idx})
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * Hb.size(0)
            if (step_idx % max(1, log_every) == 0) or (step_idx == 1):
                print({
                    "epoch": ep,
                    "step": step_idx,
                    "loss": float(loss.item()),
                    "vec": float(loss_vec.item()),
                    "kl": float(kl_val.item()) if kl_val is not None else 0.0,
                    "attn": float(loss_attn.item()) if (attn_layers_idx and attn_weight > 0.0 and kl_val is not None) else 0.0,
                    "hidden": float(loss_hidden.item()) if hidden_weight > 0.0 else 0.0,
                }, flush=True)
        train_loss = total / max(1, n_train)

        model.eval()
        with torch.no_grad():
            tot_v = 0.0
            for Hb, Mb, Eb, Tb, Fb in val_loader:
                Hb = Hb.to(device=hf.device, dtype=hf.dtype)
                Mb = Mb.to(device)
                Eb = Eb.to(device=hf.device, dtype=hf.dtype)
                pred = model(Hb, Mb)  # [B,k,d]
                pred_raw = pred
                pred_normed = pred_raw / (pred_raw.norm(dim=-1, keepdim=True) + 1e-8) * avg_norm
                try:
                    k_min = int(min(pred_normed.size(1), Eb.size(1)))
                except Exception:
                    k_min = pred_normed.size(1)
                l_vec = cosine_norm_loss(pred_normed[:, :k_min, :], Eb[:, :k_min, :]) if k_min > 0 else torch.tensor(0.0, device=device, dtype=hf.dtype)
                l_norm = (pred_raw.norm(dim=-1) - avg_norm).pow(2).mean()
                l = vec_weight * l_vec + norm_penalty * l_norm
                if kl_weight > 0.0:
                    enc_t = tok(list(Tb), return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
                    enc_f = tok(list(Fb), return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
                    prefix_ids = enc_t["input_ids"].to(device)
                    follow_ids = enc_f["input_ids"].to(device)
                    input_ids = torch.cat([prefix_ids, follow_ids], dim=1)
                    attn = torch.ones_like(input_ids)
                    hooks_t, bufs_t = _capture_attn_out(hf.model, attn_layers_idx, detach=True) if attn_layers_idx and attn_weight > 0.0 else ([], {})
                    out_t = hf.model(input_ids=input_ids, attention_mask=attn, use_cache=False, output_hidden_states=(hidden_weight > 0.0))
                    for h in hooks_t:
                        h.remove()
                    logits_t = out_t.logits
                    P = prefix_ids.size(1)
                    Lf = follow_ids.size(1)
                    sl_t = logits_t[:, P - 1 : P - 1 + Lf, :]
                    follow_emb = emb_layer(follow_ids)
                    # 検証でも前向きは正規化済みe*
                    e_pred = pred_normed  # [B,k,d]
                    full_emb = torch.cat([e_pred, follow_emb], dim=1)
                    attn_s = torch.ones(full_emb.size()[:2], dtype=torch.long, device=device)
                    hooks_s, bufs_s = _capture_attn_out(hf.model, attn_layers_idx, detach=False) if attn_layers_idx and attn_weight > 0.0 else ([], {})
                    try:
                        k = int(e_pred.size(1))
                        P_int = int(P)
                        Lf_int = int(Lf)
                        pos_ids = build_position_ids_for_virtual(P_int, Lf_int, k).to(device)
                        if pos_ids.size(0) == 1 and full_emb.size(0) > 1:
                            pos_ids = pos_ids.expand(full_emb.size(0), -1)
                        out_s = hf.model(inputs_embeds=full_emb, attention_mask=attn_s, position_ids=pos_ids, use_cache=False, output_hidden_states=(hidden_weight > 0.0))
                    except Exception:
                        out_s = hf.model(inputs_embeds=full_emb, attention_mask=attn_s, use_cache=False, output_hidden_states=(hidden_weight > 0.0))
                    for h in hooks_s:
                        h.remove()
                    logits_s = out_s.logits
                    sl_s = logits_s[:, (e_pred.size(1) - 1) : (e_pred.size(1) - 1 + Lf), :]
                    kl = _safe_kl(sl_t, sl_s, kl_temperature)
                    l = l + kl_weight * kl.float()
                    if hidden_weight > 0.0:
                        loss_hidden = hidden_match_loss(out_t, out_s, int(P), int(Lf), int(e_pred.size(1)))
                        l = l + hidden_weight * loss_hidden.float()
                tot_v += float(l.item()) * Hb.size(0)
            val_loss = tot_v / max(1, n_val)

        print({"epoch": ep, "train": train_loss, "val": val_loss, "lr": float(opt.param_groups[0]["lr"])})
        sched.step(val_loss)
        if val_loss < best:
            best = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "meta": {
                    "d_model": d,
                    "num_virtual": num_virtual,
                    "ctx_len": L,
                    "arch": arch,
                    "num_heads": num_heads,
                    "hyper_num_layers": hyper_num_layers,
                    "hyper_ffn_dim": hyper_ffn_dim,
                    "hyper_prenorm": bool(hyper_prenorm),
                    "dropout": float(dropout),
                    "vec_weight": float(vec_weight),
                    "norm_penalty": float(norm_penalty),
                }
            }, save_path)
            print("saved best to", save_path)
        # auto merge (persist newly generated stream into aggregate file)
        try:
            if (ep % max(1, int(locals().get('args', {}).get('auto_merge_every', 1))) == 0):
                pass
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=Path, required=False, default=None)
    p.add_argument("--save_path", type=Path, default=Path("artifacts/compressor_sup.pt"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl_weight", type=float, default=0.2)
    p.add_argument("--num_virtual", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "auto"])
    p.add_argument("--dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kl_temperature", type=float, default=1.0)
    p.add_argument("--hidden_weight", type=float, default=0.0)
    p.add_argument("--hidden_layer_index", type=int, default=-1)
    p.add_argument("--hidden_loss_type", type=str, default="cosine", choices=["mse", "cosine", "cosine_norm"])
    p.add_argument("--attn_weight", type=float, default=0.0)
    p.add_argument("--attn_layers", type=str, default="")
    # 追加: 層を複数回指定できる形式（--attn_layer -1 --attn_layer -2）
    p.add_argument("--attn_layer", type=int, action="append", default=None)
    p.add_argument("--attn_use_weights", action="store_true")
    p.add_argument("--resume_path", type=Path, default=None)
    # モデル選択とハイパーネット設定
    p.add_argument("--arch", type=str, default="hyper", choices=["hyper", "qattn"])
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--hyper_num_layers", type=int, default=2)
    p.add_argument("--hyper_ffn_dim", type=int, default=2048)
    p.add_argument("--hyper_prenorm", action="store_true", default=True)
    p.add_argument("--no-hyper_prenorm", dest="hyper_prenorm", action="store_false")
    # 目的関数の重み
    p.add_argument("--vec_weight", type=float, default=1.0)
    p.add_argument("--norm_penalty", type=float, default=0.02)
    # データ増分・ストリーム・生成
    p.add_argument("--data_source", type=str, default="file", choices=["file", "stream", "generate"]) 
    p.add_argument("--stream_dir", type=Path, default=None)
    p.add_argument("--data_growth_per_epoch", type=int, default=0)
    p.add_argument("--data_growth_start", type=int, default=0)
    # 生成系
    p.add_argument("--gen_per_epoch", type=int, default=0)
    p.add_argument("--gen_steps", type=int, default=60)
    p.add_argument("--gen_pairs_jsonl", type=Path, default=None)
    p.add_argument("--gen_init_mode", type=str, default="avg_embed", choices=["random", "avg_embed", "prefix_last_embed"])
    p.add_argument("--gen_ctx_len", type=int, default=64)
    p.add_argument("--gen_hidden_weight", type=float, default=0.0)
    p.add_argument("--gen_hidden_layer_index", type=int, default=-2)
    p.add_argument("--gen_hidden_loss_type", type=str, default="cosine", choices=["mse", "cosine", "cosine_norm"])
    p.add_argument("--gen_pairs_mode", type=str, default="default", choices=["default", "jsonl", "dataset", "stream"])
    p.add_argument("--gen_source_stream_dir", type=Path, default=None)
    p.add_argument("--gen_unique", action="store_true", default=True)
    p.add_argument("--no-gen_unique", dest="gen_unique", action="store_false")
    # 自動統合（各エポック末にベース集約へ追記統合）
    p.add_argument("--auto_merge_out", type=Path, default=None)
    p.add_argument("--auto_merge_every", type=int, default=1)
    args = p.parse_args()

    # attn_layersの統合（文字列と複数指定）
    combined_layers = []
    if args.attn_layers:
        try:
            combined_layers.extend([int(x.strip()) for x in args.attn_layers.split(",") if x.strip()])
        except Exception:
            pass
    if args.attn_layer:
        combined_layers.extend(args.attn_layer)
    # 重複除去と順序維持
    seen = set()
    combined_layers_unique = []
    for x in combined_layers:
        if x not in seen:
            seen.add(x)
            combined_layers_unique.append(x)
    combined_str = ",".join(str(x) for x in combined_layers_unique)

    dev = None if args.device == "auto" else torch.device(args.device)
    train(
        args.dataset_path if args.dataset_path is not None else Path(""),
        args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_weight=args.kl_weight,
        num_virtual=args.num_virtual,
        device_str=(str(dev) if dev is not None else "cpu"),
        requested_dtype=args.dtype,
        log_every=args.log_every,
        dropout=args.dropout,
        kl_temperature=args.kl_temperature,
        hidden_weight=args.hidden_weight,
        hidden_layer_index=args.hidden_layer_index,
        hidden_loss_type=args.hidden_loss_type,
        attn_weight=args.attn_weight,
        attn_layers=combined_str,
        resume_path=args.resume_path,
        attn_use_weights=args.attn_use_weights,
        arch=args.arch,
        num_heads=args.num_heads,
        hyper_num_layers=args.hyper_num_layers,
        hyper_ffn_dim=args.hyper_ffn_dim,
        hyper_prenorm=args.hyper_prenorm,
        vec_weight=args.vec_weight,
        norm_penalty=args.norm_penalty,
        data_growth_per_epoch=args.data_growth_per_epoch,
        data_growth_start=args.data_growth_start,
        data_source=args.data_source,
        stream_dir=args.stream_dir,
        gen_per_epoch=args.gen_per_epoch,
        gen_steps=args.gen_steps,
        gen_pairs_jsonl=args.gen_pairs_jsonl,
        gen_init_mode=args.gen_init_mode,
        gen_ctx_len=args.gen_ctx_len,
        gen_hidden_weight=args.gen_hidden_weight,
        gen_hidden_layer_index=args.gen_hidden_layer_index,
        gen_hidden_loss_type=args.gen_hidden_loss_type,
        gen_pairs_mode=args.gen_pairs_mode,
        gen_source_stream_dir=args.gen_source_stream_dir,
        gen_unique=args.gen_unique,
    )


if __name__ == "__main__":
    main()



