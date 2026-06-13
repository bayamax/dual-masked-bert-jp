"""recall_kit.retriever — which evicted block to pull back, two interchangeable backends.

QKIndexer    : scores in the model's OWN attention space (pre-RoPE q x pooled block keys).
               Needs a tiny pooled key summary per block (~3KB, harvested for free during
               generation). Highest accuracy (heldout top-2 ~1.0).
BGERetriever : the ELEGANT, state-free path for infinite-turn memory. Stores nothing but
               token ids; on a (rare) recall, BGE-embeds the candidate block TEXTS on the
               fly and a small learned transform bridges them to the live hidden query.
               Raw BGE cosine is ~0.25 here; the learned transform is what breaks the wall.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Indexer(nn.Module):
    """DSA-lightning-indexer-shaped (~200k params)."""
    def __init__(self, L, Hq, Hkv, D, d=32):
        super().__init__()
        self.group = Hq // Hkv
        self.Aq = nn.Parameter(torch.zeros(L, Hq, D, d))
        self.Bk = nn.Parameter(torch.zeros(L, Hkv, 2 * D, d))
        self.w = nn.Parameter(torch.zeros(L, Hq))

    def forward(self, q, k):                       # q:[L,Hq,D]  k:[nB,L,Hkv,2D] -> [nB]
        qp = torch.einsum("lhd,lhde->lhe", q, self.Aq)
        kp = torch.einsum("blkf,lkfe->blke", k, self.Bk).repeat_interleave(self.group, 2)
        sim = F.relu(torch.einsum("lhe,blhe->blh", qp, kp))
        return torch.einsum("blh,lh->b", sim, torch.softmax(self.w.view(-1), 0).view(self.w.shape))


class QKIndexer:
    def __init__(self, model, dims, device="cpu"):
        self.model, self.device = model, device

    @classmethod
    def load(cls, path, dims, device="cpu"):
        z = np.load(path)
        Hq, Hkv, D = dims; L = z["Aq"].shape[0]
        m = Indexer(L, Hq, Hkv, D)
        m.load_state_dict({"Aq": torch.tensor(z["Aq"]), "Bk": torch.tensor(z["Bk"]),
                           "w": torch.tensor(z["w"])})
        obj = cls(m.to(device).eval(), dims, device)
        return obj

    @torch.no_grad()
    def rank(self, q_LHqD, keys_nBLHkv2D):
        q = torch.as_tensor(q_LHqD, dtype=torch.float32, device=self.device)
        k = torch.as_tensor(keys_nBLHkv2D, dtype=torch.float32, device=self.device)
        return self.model(q, k).argsort(descending=True).tolist()


class BGEHead(nn.Module):
    def __init__(self, qdim, kdim, d=128):
        super().__init__()
        self.Wq = nn.Sequential(nn.Linear(qdim, d), nn.GELU(), nn.Linear(d, d))
        self.Wk = nn.Sequential(nn.Linear(kdim, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, q, k):                       # q:[qdim] k:[nB,kdim] -> [nB]
        return self.Wq(q) @ self.Wk(k).T


class BGERetriever:
    """State-free: keeps only token ids; embeds candidate block texts with BGE on demand."""
    def __init__(self, head, bge, device="cpu"):
        self.head, self.bge, self.device = head, bge, device

    @classmethod
    def load(cls, path, bge, device="cpu"):
        z = np.load(path)
        qdim, kdim, d = int(z["qdim"]), int(z["kdim"]), int(z["d"])
        head = BGEHead(qdim, kdim, d)
        sd = {k: torch.tensor(z[k]) for k in z.files if k not in ("qdim", "kdim", "d")}
        head.load_state_dict(sd)
        return cls(head.to(device).eval(), bge, device)

    @torch.no_grad()
    def rank(self, q_flat, block_texts):
        bk = self.bge.encode(list(block_texts), normalize_embeddings=True,
                             show_progress_bar=False)
        q = torch.as_tensor(np.asarray(q_flat, np.float32).ravel(), device=self.device)
        k = torch.as_tensor(np.asarray(bk, np.float32), device=self.device)
        return self.head(q, k).argsort(descending=True).tolist()

    @torch.no_grad()
    def rank_raw(self, q_flat, block_texts):
        """No-transform baseline: BGE(query-text) cosine. Provided for comparison only."""
        bk = self.bge.encode(list(block_texts), normalize_embeddings=True,
                             show_progress_bar=False)
        # caller must pass a BGE query vector as q_flat here
        return list(np.argsort(-(bk @ np.asarray(q_flat, np.float32))))
