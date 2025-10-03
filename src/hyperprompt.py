from __future__ import annotations

import torch
import torch.nn as nn


class HyperPromptNet(nn.Module):
    """ハイパーネットワーク型のソフトプロンプト生成器

    - 学習可能な k 個のクエリ埋め込みから文脈隠れ状態にクロスアテンション
    - 段（num_layers）を重ね、各段で MHA + FFN の残差ブロック
    - 最終的に [B, k, d] の仮想トークン埋め込み列を出力
    - マスクは key_padding_mask として使用（True=pad）
    """

    def __init__(
        self,
        d_model: int,
        num_virtual: int = 1,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        prenorm: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_virtual = num_virtual
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.prenorm = prenorm

        self.query_embed = nn.Parameter(torch.randn(num_virtual, d_model))
        nn.init.xavier_uniform_(self.query_embed)

        self.attn_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.ln_q_attn = nn.ModuleList()
        self.ln_q_ffn = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        for _ in range(num_layers):
            self.attn_blocks.append(
                nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
            )
            self.ffn_blocks.append(
                nn.Sequential(
                    nn.Linear(d_model, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, d_model),
                )
            )
            self.ln_q_attn.append(nn.LayerNorm(d_model))
            self.ln_q_ffn.append(nn.LayerNorm(d_model))

        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, ctx_states: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # ctx_states: [B, L, d]
        batch_size, _, _ = ctx_states.size()
        q = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        q = q.to(dtype=ctx_states.dtype, device=ctx_states.device)
        # Inference-safe: clone K/V to avoid using inference tensors in autograd path
        k = ctx_states.contiguous().clone()

        kpm = None
        if attn_mask is not None:
            kpm = (attn_mask == 0)

        for li in range(self.num_layers):
            attn = self.attn_blocks[li]
            ffn = self.ffn_blocks[li]
            if self.prenorm:
                q_norm = self.ln_q_attn[li](q)
                attn_out, _ = attn(query=q_norm, key=k, value=k, key_padding_mask=kpm, need_weights=False)
                q = q + self.drop(attn_out)
                q_ffn = self.ln_q_ffn[li](q)
                q = q + self.drop(ffn(q_ffn))
            else:
                attn_out, _ = attn(query=q, key=k, value=k, key_padding_mask=kpm, need_weights=False)
                q = self.ln_q_attn[li](q + self.drop(attn_out))
                q = self.ln_q_ffn[li](q + self.drop(ffn(q)))

        return self.final_ln(q)



