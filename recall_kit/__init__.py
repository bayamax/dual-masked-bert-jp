"""recall_kit — per-position recall for SP-compressed, infinite-turn memory.

GATE detects (from the live hidden state, no stored state) the rare moment a decode
position needs a fact evicted past the exposure window; a RETRIEVER then pulls one block,
either in the model's QK space (QKIndexer, ~3KB key summary/block) or fully state-free
(BGERetriever: only token ids persist, BGE embeds candidate blocks on demand + a tiny
learned transform). Trained & validated on real Dolphin-R1 long CoTs.
"""
from .gate import RecallGate, LAYERS
from .retriever import QKIndexer, BGERetriever, Indexer, BGEHead
from .runtime import RecallRuntime

__all__ = ["RecallGate", "QKIndexer", "BGERetriever", "RecallRuntime",
           "Indexer", "BGEHead", "LAYERS"]
