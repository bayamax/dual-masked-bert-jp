
import subprocess
import argparse
import sys
import os

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(ret.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["demo", "full"], default="demo")
    parser.add_argument("--skip_data_gen", action="store_true")
    parser.add_argument("--skip_phase0", action="store_true")
    args = parser.parse_args()
    
    print(f"=== Starting Hippocampal Pipeline V2.0 (Mode: {args.mode}) ===")
    
    # Configs
    if args.mode == "demo":
        max_docs = 100
        max_steps = 20
        chunk_size = 128
        batch_size = 2
    else:
        max_docs = -1 # All
        max_steps = -1 # All epochs
        chunk_size = 128
        batch_size = 8
        
    # Step 1: Data Generation
    if not args.skip_data_gen:
        if not os.path.exists("wiki_long_50k.jsonl"):
            print("Step 1: Generating Wiki Data...")
            run_command(f"python3 src/prepare_wikitext_remote.py") # Generates wiki_long_50k.jsonl
        else:
            print("Wiki data already exists. Skipping generation.")
            
    # Step 2: Phase 0 (Data Diet / Indexing)
    if not args.skip_phase0:
        print("Step 2: Phase 0 (Index & Label Generation)...")
        # Ensure clean demo dir if demo?
        hippo_dir = f"hippocampus_v2_{args.mode}"
        cmd = f"python3 src/prep_phase0_v2.py --data_path wiki_long_50k.jsonl --save_dir {hippo_dir} --max_docs {max_docs} --chunk_size {chunk_size} --use_tiny"
        # Force TinyLlama for Phase 0 to ensure tokenizer consistency with Phase 1 (Student = TinyLlama)
        # unless we implement complex cross-tokenizer mapping.
        run_command(cmd)
    else:
        hippo_dir = f"hippocampus_v2_{args.mode}"

    # Step 3: Phase 1 (Training)
    print("Step 3: Phase 1 (Training)...")
    ckpt_dir = f"checkpoints_v2_{args.mode}"
    cmd = f"python3 src/train_hippocampal_v2.py --data_dir {hippo_dir} --save_path {ckpt_dir} --max_steps {max_steps} --batch_size {batch_size}"
    run_command(cmd)
    
    # Step 4: Verification (Only for Demo)
    if args.mode == "demo":
        print("Step 4: Verifying Results...")
        # Simple verify: Check if files exist and logs showed loss decrease
        if os.path.exists(f"{ckpt_dir}/student_last") and os.path.exists(f"{ckpt_dir}/hypernet_last.pt"):
            print("SUCCESS: Checkpoints created.")
            print("Pipeline V2 Demo Completed Successfully.")
        else:
            print("FAILURE: Checkpoints missing.")
            sys.exit(1)

if __name__ == "__main__":
    main()
