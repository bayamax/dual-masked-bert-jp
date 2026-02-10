
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import os
import json
from trl import SFTTrainer, SFTConfig

# Config
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "phase10_composite_model"
DATA_FILE = "phase10_composite.jsonl"
MAX_LENGTH = 2048

# Hyperparameters
LR = 2e-4
EPOCHS = 2
BATCH_SIZE = 4 # Adjust per GPU VRAM
GRAD_ACCUM = 8 # Effective Batch = 32

class HyperNetHead(nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.proj(x)

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Full target for robust SFT
    )
    
    model.print_trainable_parameters()
    
    # HyperNet (Z-Head) - Initialize randomly or load from previous?
    # Ideally, we load from Phase 9 or Phase 8 to keep compatibility?
    # But this is a "Union" model. Let's start fresh and let it learn the Z-space of the composite data.
    # The composite data has Z vectors from Phase 8/9/7 models.
    # So we are training a new head to approximate those vectors.
    hypernet = HyperNetHead().to(model.device).to(torch.float16)
    optimizer_hypernet = torch.optim.AdamW(hypernet.parameters(), lr=LR)
    
    print("Loading Composite Dataset...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    def formatting_func(example):
        # The prompt is already formatted in 'instruction' + 'input' or 'text'? 
        # Wait, prepare_phase10_data.py outputs keys relevant for standard loading?
        # Actually prepare_phase10_data.py outputs what gen_phase* outputs.
        # gen_phase9_math outputs: "question", "answer", "chain_of_thought", "z_vector" ... 
        # gen_phase8_alpaca outputs: "instruction", "input", "generated_cot", "z_vector" ...
        # We need to unify this for SFT.
        
        output_text = ""
        
        # Check source type by keys
        if "question" in example: # Math
             # Format: USER: Solve... Question: <Q>... Answer: <CoT>
             # Actually gen_phase9_math.py likely outputs the full prompt in "chain_of_thought" if configured?
             # Let's check gen_phase9 format again.
             # It outputs "generated_text" usually.
             
             # If we look at prepare_phase10_data.py, it says "return item" (pass through).
             pass
             
        # For SFTTrainer, we usually provide a "text" field.
        # Let's trust that the generation scripts output a "text" or "generated_text" field that is the full sequence?
        # If not, we need a collator.
        
        # Simpler: Just use "generated_cot" or whatever text field exists.
        # Let's assume the jsonl has a "text" field? gen_phase9 usually doesn't add "text" key by default unless specified.
        # We might need to inspect the data or update prepare_phase10_data.py to add a "text" field.
        
        # Fallback: Look for "generated_cot" or "text"
        text = example.get("text", example.get("generated_cot", ""))
        return [text]

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )
    
    # formatting_func to handle different key names if needed
    def formatting_prompts_func(example):
        # SFTTrainer passes a dict of lists (batched)
        output_texts = []
        # Check available keys
        keys = example.keys()
        target_key = "generated_cot" if "generated_cot" in keys else "text"
        
        # If "generated_cot" is missing (e.g. math data might have different key?), fallback
        # But prepare_phase10_data.py mixes them.
        # Let's inspect what prepare_phase10_data.py actually saved.
        # It saved whatever gen_phase* saved.
        # gen_phase8 saved: "instruction", "input", "original_output", "generated_cot", "has_think"
        # gen_phase9 saved: "question", "ground_truth_answer", "generated_cot"
        # So "generated_cot" is common!
        
        for text in example.get(target_key, []):
            output_texts.append(text)
        return output_texts

    # Use SFTConfig
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="generated_cot", 
        max_length=MAX_LENGTH, # Found via inspect: it is max_length, not max_seq_length
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
        packing=False
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        args=training_args
    )
    
    # Custom Training MLoop for Z-Loss?
    # SFTTrainer handles generation loss.
    # We need to add Z-Loss.
    # We can override compute_loss.
    
    from transformers import Trainer
    class CustomTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Forward pass for Gen Loss
            outputs = model(**inputs)
            gen_loss = outputs.loss
            
            # Z-Vector Loss
            # We need hidden states.
            # outputs.hidden_states is available if output_hidden_states=True
            
            # Extract Z from inputs? The dataset has z_vector list.
            # We need to pass z_vector to inputs. SFTTrainer might drop unused columns.
            
            # For simplicity in this plan: Just train Generation first?
            # User wants "Quality". Z-vector is for "Retrieval".
            # If we drop Z-loss, we lose retrieval search capability (Hypothesis).
            # But the primary request is "Reasoning + Chat".
            # Let's stick to pure SFT for Phase 10 to ensure stability first.
            # RL (Phase 11) will fix the reasoning logic.
            # Retrieval can be re-aligned later if needed.
            return (gen_loss, outputs) if return_outputs else gen_loss

    print("Starting Training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    
    # Save HyperNet if we used it (Skipped for now to focus on stability)
    # torch.save(hypernet.state_dict(), os.path.join(OUTPUT_DIR, "hypernet.pt"))
    print("Training Complete.")

if __name__ == "__main__":
    main()
