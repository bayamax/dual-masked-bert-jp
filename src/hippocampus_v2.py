
import torch
import torch.nn.functional as F
import json
import os
import numpy as np

class HippocampusV2:
    """
    Hippocampus v2.0: Sparse, Quantized, and Text-Based.
    Stores:
        - Z-vectors: Int8 Quantized in RAM (or mmap).
        - Text: On disk (JSONL) with offset index.
        - Metadata: {Token_Index: Target_ID} for Sparse Labels.
    """
    def __init__(self, d_model=512, storage_dir="hippocampus_v2"):
        self.d_model = d_model
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        self.z_path = os.path.join(storage_dir, "z_vectors.bin")
        self.idx_path = os.path.join(storage_dir, "text_offsets.npy")
        self.text_path = os.path.join(storage_dir, "corpus.jsonl")
        self.labels_path = os.path.join(storage_dir, "labels.pt")
        
        self.z_vectors = [] # List of Int8 Tensors (or appended)
        self.text_offsets = []
        self.text_file = None
        self.labels = {} # {current_idx: target_idx}
        
    def add(self, z_vec, text_tokens):
        """
        Add a memory trace.
        z_vec: [D] Float32 -> Converted to Int8
        text_tokens: Tensor or List -> Saved as Text String (or Tokens)
        """
        # Quantize Z: Float32 [-1, 1] -> Int8 [-127, 127]
        # Simple scaling: * 127
        z_int8 = (z_vec * 127.0).clamp(-127, 127).to(torch.int8)
        self.z_vectors.append(z_int8.cpu())
        
        # Save Text
        if self.text_file is None:
            self.text_file = open(self.text_path, 'a+')
            
        # Record Offset
        self.text_file.seek(0, os.SEEK_END)
        offset = self.text_file.tell()
        self.text_offsets.append(offset)
        
        # Write
        # Assuming text_tokens is already decoded string OR list of ids
        # We store list of IDs for exact reconstruction
        entry = {"ids": text_tokens.tolist()}
        self.text_file.write(json.dumps(entry) + "\n")
        self.text_file.flush() # Ensure detailed consistency
        
        return len(self.z_vectors) - 1

    def add_label(self, current_idx, target_idx):
        self.labels[current_idx] = target_idx

    def finalize(self):
        """Save everything to disk for efficient loading"""
        if self.text_file:
            self.text_file.close()
            
        # Stack Z and Save
        if self.z_vectors:
            z_stack = torch.stack(self.z_vectors)
            torch.save(z_stack, self.z_path)
            
        # Save Offsets
        np.save(self.idx_path, np.array(self.text_offsets, dtype=np.int64))
        
        # Save Labels
        torch.save(self.labels, self.labels_path)
        print(f"Hippocampus Saved: {len(self.z_vectors)} items.")
        
    def load(self, device="cpu"):
        """Load Z into RAM/GPU"""
        if os.path.exists(self.z_path):
            self.z_bank = torch.load(self.z_path, map_location="cpu") # Keep Int8 on CPU mostly? Or GPU?
            # If GPU VRAM is tight, keep CPU. But for search, we need GPU.
            # Int8 takes 1/4 of Float32. 50k items * 512 bytes = 25MB. Tiny.
            # Even 1M items = 500MB.
            # We can put on GPU.
            self.z_bank = self.z_bank.to(device)
            
        if os.path.exists(self.idx_path):
            self.text_offsets = np.load(self.idx_path)
            
        if os.path.exists(self.labels_path):
            self.labels = torch.load(self.labels_path)
            
        self.text_file = open(self.text_path, 'r')
        self.device = device
        
    def get_text(self, idx):
        """Retrieve Raw Tokens"""
        offset = self.text_offsets[idx]
        self.text_file.seek(offset)
        line = self.text_file.readline()
        data = json.loads(line)
        return torch.tensor(data["ids"])

    def get_z(self, idx):
        """Retrieve Dequantized Z"""
        z_int8 = self.z_bank[idx].float()
        return z_int8 / 127.0
        
    def search(self, query_z, top_k=1):
        """
        Search for nearest Z.
        query_z: [B, D] Float
        """
        # Dequantize Bank on the fly? Or Matmul Int8? 
        # For simplicity/precision, Dequantize to BFloat16/Float32
        bank_float = self.z_bank.float() / 127.0
        
        # Cosine Sim
        # Norms
        q_norm = F.normalize(query_z, p=2, dim=1)
        b_norm = F.normalize(bank_float, p=2, dim=1)
        
        scores = torch.mm(q_norm, b_norm.t())
        return scores.topk(top_k, dim=1)
