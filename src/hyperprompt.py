from __future__ import annotations

import torch
import torch.nn as nn



class ReconstructionDecoder(nn.Module):
    """
    Recursive Auto-Encoder Decoder: 
    Reconstructs BOTH the Previous Z (Memory) and Current Chunk Tokens (Content).
    """
    def __init__(self, d_model: int, vocab_size: int, chunk_len: int, z_len: int, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.chunk_len = chunk_len # 28
        self.z_len = z_len         # 4
        
        # Queries for Z reconstruction (4) + Text reconstruction (28)
        self.total_queries_len = z_len + chunk_len
        self.output_queries = nn.Parameter(torch.randn(self.total_queries_len, d_model))
        nn.init.xavier_uniform_(self.output_queries)
        
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        
        # Heads
        self.proj_z = nn.Linear(d_model, d_model)      # Reconstruct Z vectors
        self.proj_text = nn.Linear(d_model, vocab_size) # Reconstruct Token IDs

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: [B, k, d]
        B = z.size(0)
        # q: [B, Z+L, d]
        q = self.output_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-Attend: Queries attend to Current Z
        attn_out, _ = self.attn(query=q, key=z, value=z)
        # Upcast for BF16
        x = self.ln1((q + attn_out).float()).to(dtype=q.dtype)
        
        # FFN
        out = self.ffn(x)
        x = self.ln2((x + out).float()).to(dtype=x.dtype)
        
        # Split outputs
        # First z_len tokens -> Z reconstruction
        # Next chunk_len tokens -> Text reconstruction
        x_z = x[:, :self.z_len, :]
        x_text = x[:, self.z_len:, :]
        
        z_rec = self.proj_z(x_z)        # [B, 4, d]
        text_logits = self.proj_text(x_text) # [B, 28, V]
        
        return z_rec, text_logits

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
        # Reconstruction Config
        use_decoder: bool = False,
        vocab_size: int = 32000,
        chunk_len: int = 28,
        output_dim: int = None, # New param
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_virtual = num_virtual
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.prenorm = prenorm
        self.use_decoder = use_decoder

        self.query_embed = nn.Parameter(torch.randn(num_virtual, d_model))
        nn.init.xavier_uniform_(self.query_embed)
        
        # Output Projection if needed
        if output_dim is not None and output_dim != d_model:
            self.out_proj = nn.Linear(d_model, output_dim)
        else:
            self.out_proj = None

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
        
        if use_decoder:
            # Phase 1' Projector: HyperNet -> Linear -> LayerNorm -> Llama
            self.projector = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            )
            # Legacy decoder removed
            self.decoder = None

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
                # Upcast to float32 for LayerNorm stability (BF16 issue)
                # DEBUG PRINT

                if q.dtype == torch.bfloat16:
                   pass # This should be fine if we cast
                q_norm = self.ln_q_attn[li](q)
                attn_out, _ = attn(query=q_norm, key=k, value=k, key_padding_mask=kpm, need_weights=False)
                q = q + self.drop(attn_out)
                q_ffn = self.ln_q_ffn[li](q)
                q = q + self.drop(ffn(q_ffn))
            else:
                attn_out, _ = attn(query=q, key=k, value=k, key_padding_mask=kpm, need_weights=False)
                # BF16 fix for Post-Norm
                res = q + self.drop(attn_out)
                q = self.ln_q_attn[li](res.float()).to(dtype=res.dtype)
                
                res_ffn = q + self.drop(ffn(q))
                q = self.ln_q_ffn[li](res_ffn.float()).to(dtype=res_ffn.dtype)

        out = self.final_ln(q)
        
        if self.use_decoder and hasattr(self, 'projector'):
            # Projector contains LN, so cast to float input
            out = self.projector(out)
            
        if self.out_proj is not None:
            out = self.out_proj(out)
            
        return out

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Decoder has been removed in Phase 1'")



