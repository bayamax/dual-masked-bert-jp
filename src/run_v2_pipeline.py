
import argparse
import subprocess
import os
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(ret)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["demo", "full"], default="demo")
    parser.add_argument("--skip_data_gen", action="store_true")
    parser.add_argument("--skip_phase0", action="store_true")
    parser.add_argument("--skip_phase1", action="store_true")
    args = parser.parse_args()
    
    print(f"=== Starting Hippocampal Pipeline V2.0 (Mode: {args.mode}) ===")
    
    # Configuration
    if args.mode == "demo":
        max_docs = 100
        max_steps = 20
        chunk_size = 128
        batch_size = 2
    else:
        max_docs = -1 # All
        max_steps = -1 # All epochs
        chunk_size = 128
        batch_size = 32 # Safe for RTX 4090
        
    # Step 1: Data Generation
    if not args.skip_data_gen:
        if os.path.exists("wiki_long_50k.jsonl"):
            print("Wiki data already exists. Skipping generation.")
        else:
            print("Step 1: Generating Wiki Data...")
            run_command("python3 src/prepare_wikitext_remote.py")
    
    # Step 2: Phase 0 (Teacher Processing)
    if not args.skip_phase0:
        print("Step 2: Phase 0 (Index & Label Generation)...")
        # Ensure clean demo dir if demo?
        hippo_dir = f"hippocampus_v2_{args.mode}"
        cmd = f"python3 src/prep_phase0_v2.py --data_path wiki_long_50k.jsonl --save_dir {hippo_dir} --max_docs {max_docs} --chunk_size {chunk_size} --use_tiny"
        # Force TinyLlama for Phase 0 to ensure tokenizer consistency with Phase 1 (Student = TinyLlama)
        run_command(cmd)
    else:
        hippo_dir = f"hippocampus_v2_{args.mode}"
        
    # Step 3: Phase 1 (Training)
    if not args.skip_phase1:
        print("Step 3: Phase 1 (Training)...")
        save_dir = f"checkpoints_v2_{args.mode}"
        cmd = f"python3 src/train_hippocampal_v2.py --data_dir {hippo_dir} --save_path {save_dir} --max_steps {max_steps} --batch_size {batch_size}"
        run_command(cmd)
        
    # Verification (Demo Only)
    if args.mode == "demo":
        print("Step 4: Verifying Results...")
        if os.path.exists(f"checkpoints_v2_{args.mode}"):
            print("SUCCESS: Checkpoints created.")
        else:
            print("FAILURE: No checkpoints found.")
            sys.exit(1)
            
    print(f"Pipeline V2 {args.mode} Completed Successfully.")

if __name__ == "__main__":
    main()
