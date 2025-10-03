from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .hf_model import load_tokenizer_and_model


@torch.no_grad()
def compute_avg_input_norm(hf) -> float:
	emb = hf.model.get_input_embeddings().weight.detach().to(torch.float32)
	return emb.norm(dim=-1).mean().item()


def build_position_ids_for_virtual(prefix_len: int, follow_len: int, num_virtual: int, strategy: str = "tail") -> torch.Tensor:
	# prefix_len: 教師のプレフィックス長（BOS含む）
	# virtualは教師の末尾位置に合わせる。k個なら [prefix_len - k, ..., prefix_len - 1]
	if strategy not in ("tail",):
		raise ValueError("unsupported strategy")
	start = max(0, prefix_len - num_virtual)
	virt_pos = torch.arange(start, start + num_virtual, dtype=torch.long)
	follow_pos = torch.arange(prefix_len, prefix_len + follow_len, dtype=torch.long)
	return torch.cat([virt_pos, follow_pos], dim=0).unsqueeze(0)  # [1, k+L]


@torch.inference_mode()
def get_teacher_targets(
	hf,
	prefix_ids: torch.Tensor,
	follow_ids: torch.Tensor,
	temperature: float = 1.0,
	hidden_layer_index: int = -1,
):
	# teacher: [prefix, follow]
	input_ids = torch.cat([prefix_ids, follow_ids], dim=1)
	attn = torch.ones_like(input_ids)
	out = hf.model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
	logits = out.logits  # [1, Ltot, V]
	hidden = out.hidden_states[hidden_layer_index]  # [1, Ltot, d]
	P = prefix_ids.size(1)
	Lf = follow_ids.size(1)
	# 次トークン分布（follow各トークンに対する予測）: 直前位置のlogits
	sl = logits[:, P - 1 : P - 1 + Lf, :]  # [1, Lf, V]
	logprobs = F.log_softmax(sl / temperature, dim=-1)
	# 隠れ状態ターゲット（同じ直前位置）
	h_slice = hidden[:, P - 1 : P - 1 + Lf, :]  # [1, Lf, d]
	return logprobs, h_slice


def optimize_virtual(
	model_name: str,
	text: str,
	follow_text: str,
	num_virtual: int = 1,
	steps: int = 300,
	lr: float = 0.1,
	temperature: float = 1.0,
	lambda_norm: float = 0.05,
	seed: int = 42,
	save_path: Path | None = None,
	device_override: str | None = None,
	init_mode: str = "random",
	hidden_weight: float = 0.0,
	hidden_layer_index: int = -1,
	hidden_loss_type: str = "mse",
    hf_obj: object | None = None,
    log_every: int = 0,
):
	torch.manual_seed(seed)
	hf = hf_obj if hf_obj is not None else load_tokenizer_and_model(model_name)
	tok = hf.tokenizer
	device = hf.device
	if device_override in ("cpu", "mps") and hf_obj is None:
		device = torch.device(device_override)
		# CPUではfloat32へキャストして勾配計算を安定化
		if device.type == "cpu":
			hf.model.to(device=device, dtype=torch.float32)
		else:
			hf.model.to(device)

	# 教師シーケンス
	pref = tok(text, return_tensors="pt", add_special_tokens=True)
	prefix_ids = pref["input_ids"].to(device)
	follow = tok(follow_text, return_tensors="pt", add_special_tokens=False)
	follow_ids = follow["input_ids"].to(device)

	# 目標（教師）: evalで固定
	hf.model.eval()
	with torch.no_grad():
		teacher_logprobs, teacher_hidden = get_teacher_targets(
			hf, prefix_ids, follow_ids, temperature=temperature, hidden_layer_index=hidden_layer_index
		)
		teacher_probs = teacher_logprobs.exp().to(device).detach().clone()
		teacher_hidden = teacher_hidden.to(device).detach().clone()

	# 初期ログ（学習開始合図）
	print({
		"status": "init",
		"device": str(device),
		"model_dtype": str(hf.dtype),
		"steps": int(steps),
		"lr": float(lr),
		"num_virtual": int(num_virtual),
		"prefix_len": int(prefix_ids.size(1)),
		"follow_len": int(follow_ids.size(1)),
		"hidden_weight": float(hidden_weight),
		"hidden_layer_index": int(hidden_layer_index),
		"hidden_loss_type": hidden_loss_type,
	}, flush=True)

	# 学習対象: 仮想トークン e*（k, d）
	emb_layer = hf.model.get_input_embeddings()
	d_model = emb_layer.weight.shape[1]
	avg_norm = compute_avg_input_norm(hf)

	# 初期化
	init_mode = init_mode.lower()
	if init_mode not in ("random", "avg_embed", "prefix_last_embed"):
		raise ValueError(f"unknown init_mode: {init_mode}")

	if init_mode == "random":
		init_e = torch.randn((num_virtual, d_model), device=device, dtype=emb_layer.weight.dtype)
	elif init_mode == "avg_embed":
		mean_vec = emb_layer.weight.detach().to(device=device, dtype=emb_layer.weight.dtype).mean(dim=0)
		init_e = mean_vec.unsqueeze(0).repeat(num_virtual, 1)
	else:  # prefix_last_embed
		last_id = prefix_ids[0, -1].item()
		last_vec = emb_layer.weight.detach()[last_id].to(device=device, dtype=emb_layer.weight.dtype)
		init_e = last_vec.unsqueeze(0).repeat(num_virtual, 1)

	# 初期ノルムを平均へ揃える
	init_e = init_e / (init_e.norm(dim=-1, keepdim=True) + 1e-8) * avg_norm
	e_star = torch.nn.Parameter(init_e)

	# e* 初期統計
	print({
		"status": "e_star_init",
		"e_star_norm_mean": float(init_e.norm(dim=-1).mean().item()),
		"avg_norm": float(avg_norm),
	}, flush=True)

	opt = torch.optim.Adam([e_star], lr=lr)
	# モデルの勾配は不要・推論挙動で固定
	hf.model.requires_grad_(False)
	hf.model.eval()

	attn = torch.ones((1, num_virtual + follow_ids.size(1)), dtype=torch.long, device=device)

	for t in range(1, steps + 1):
		opt.zero_grad(set_to_none=True)

		with torch.enable_grad():
			follow_emb = emb_layer(follow_ids)  # [1, Lf, d]
			virt = e_star.unsqueeze(0)  # [1, k, d]
			full_emb = torch.cat([virt, follow_emb], dim=1)  # [1, k+Lf, d]

			# 位置を教師と整合させる（virtualは教師prefixの末尾位置に割当）
			P = int(prefix_ids.size(1))
			Lf = int(follow_ids.size(1))
			pos_ids = build_position_ids_for_virtual(P, Lf, num_virtual).to(device)

			try:
				out_s = hf.model(
					inputs_embeds=full_emb,
					attention_mask=attn,
					position_ids=pos_ids,
					use_cache=False,
					output_hidden_states=(hidden_weight > 0.0),
				)
			except IndexError:
				# 一部transformers実装ではposition_idsの最大値がシーケンス長未満である必要がある
				# その場合は相対位置にフォールバック（[0..k-1, k..k+Lf-1]）
				out_s = hf.model(
					inputs_embeds=full_emb,
					attention_mask=attn,
					use_cache=False,
					output_hidden_states=(hidden_weight > 0.0),
				)
			logits_s = out_s.logits.clone()  # [1, k+Lf, V]
			# 生徒側の予測スライス（follow分の次トークン）
			sl_s = logits_s[:, num_virtual - 1 : num_virtual - 1 + follow_ids.size(1), :]
			logprobs_s = F.log_softmax(sl_s / temperature, dim=-1).clone()

			# KL(p || q) = sum p*(log p - log q) -> - sum p*log q + const
			# 勾配計算対象のlogprobs_sと演算する前に、teacher_probsは完全に独立なテンソル
			loss_ce = -(teacher_probs * logprobs_s).sum(dim=-1).mean()
			# 隠れ状態一致
			if hidden_weight > 0.0:
				h_s = out_s.hidden_states[hidden_layer_index]
				h_s_slice = h_s[:, num_virtual - 1 : num_virtual - 1 + follow_ids.size(1), :]
				# cosine / cosine_norm / mse
				if hidden_loss_type == "mse":
					loss_h = F.mse_loss(h_s_slice, teacher_hidden.to(dtype=h_s_slice.dtype))
				elif hidden_loss_type == "cosine":
					pred_n = F.normalize(h_s_slice, dim=-1)
					targ_n = F.normalize(teacher_hidden.to(dtype=h_s_slice.dtype), dim=-1)
					loss_h = (1.0 - (pred_n * targ_n).sum(dim=-1)).mean()
				elif hidden_loss_type == "cosine_norm":
					pred_n = F.normalize(h_s_slice, dim=-1)
					targ_n = F.normalize(teacher_hidden.to(dtype=h_s_slice.dtype), dim=-1)
					cos = (pred_n * targ_n).sum(dim=-1)
					norm_l2 = (h_s_slice.norm(dim=-1) - teacher_hidden.to(dtype=h_s_slice.dtype).norm(dim=-1)).pow(2)
					loss_h = (1.0 - cos + 0.1 * norm_l2).mean()
				else:
					raise ValueError(f"unknown hidden_loss_type: {hidden_loss_type}")
			else:
				loss_h = torch.tensor(0.0, device=device, dtype=logprobs_s.dtype)
			# ノルム整合
			norm_diff = (e_star.norm(dim=-1) - avg_norm).pow(2).mean()
			loss = loss_ce + lambda_norm * norm_diff + hidden_weight * loss_h
		loss.backward()
		opt.step()

		with torch.no_grad():
			n = e_star.norm(dim=-1, keepdim=True) + 1e-8
			e_star.copy_(e_star / n * avg_norm)

		should_log = (log_every and (t % max(1, log_every) == 0)) or (t == 1) or (t == steps)
		if should_log:
			print({"step": t, "loss": float(loss.item()), "ce": float(loss_ce.item()), "norm_pen": float((lambda_norm * norm_diff).item())}, flush=True)

	result = {
		"e_star": e_star.detach().to(torch.float32).cpu(),
		"meta": {
			"model_name": model_name,
			"num_virtual": num_virtual,
			"prefix_len": int(prefix_ids.size(1)),
			"follow_len": int(follow_ids.size(1)),
			"avg_norm": float(avg_norm),
			"temperature": float(temperature),
			"hidden_weight": float(hidden_weight),
			"hidden_layer_index": int(hidden_layer_index),
			"hidden_loss_type": hidden_loss_type,
		},
		"text": text,
		"follow_text": follow_text,
	}
	if save_path is not None:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(result, save_path)
		print("saved:", save_path, flush=True)
	return result


def main() -> None:
	p = argparse.ArgumentParser()
	p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
	p.add_argument("--text", type=str, required=True)
	p.add_argument("--follow_text", type=str, default="")
	p.add_argument("--num_virtual", type=int, default=1)
	p.add_argument("--steps", type=int, default=300)
	p.add_argument("--lr", type=float, default=0.1)
	p.add_argument("--temperature", type=float, default=1.0)
	p.add_argument("--lambda_norm", type=float, default=0.05)
	p.add_argument("--save_path", type=Path, default=Path("artifacts/e_star.pt"))
	p.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "auto"])
	p.add_argument("--init_mode", type=str, default="random", choices=["random", "avg_embed", "prefix_last_embed"])
	p.add_argument("--hidden_weight", type=float, default=0.0)
	p.add_argument("--hidden_layer_index", type=int, default=-1)
	p.add_argument("--hidden_loss_type", type=str, default="mse", choices=["mse", "cosine", "cosine_norm"])
	p.add_argument("--log_every", type=int, default=0)
	args = p.parse_args()

	optimize_virtual(
		model_name=args.model_name,
		text=args.text,
		follow_text=args.follow_text,
		num_virtual=args.num_virtual,
		steps=args.steps,
		lr=args.lr,
		temperature=args.temperature,
		lambda_norm=args.lambda_norm,
		save_path=args.save_path,
		device_override=(None if args.device == "auto" else args.device),
		init_mode=args.init_mode,
		hidden_weight=args.hidden_weight,
		hidden_layer_index=args.hidden_layer_index,
		hidden_loss_type=args.hidden_loss_type,
		hf_obj=None,
		log_every=args.log_every,
	)


if __name__ == "__main__":
	main()


