"""
Phase 7 SFT Data Preparation (for Reference Injection)

Takes the raw CoT data and formats it for SFT training.
Format: [参照情報: {answer_context}]\nUser: {question}\nModel: {correct_answer}
"""

import json
import os
import re

INPUT_FILE = "phase7_raw_cot.jsonl"
OUTPUT_FILE = "phase7_sft_injection.jsonl"

def extract_final_answer(cot_text):
    """Extract the final answer from CoT text."""
    # Look for boxed answer first
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', cot_text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for "Final Answer:" pattern
    final_match = re.search(r'Final Answer[:\s]+(.+?)(?:\n|$)', cot_text, re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()
    
    # Look for "= X" at end of solution
    eq_match = re.search(r'=\s*(\d+)', cot_text)
    if eq_match:
        return eq_match.group(1)
    
    return None

def main():
    print(f"Processing {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found")
        return
    
    samples = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                question = data['question']
                cot = data['generated_cot']
                
                # Extract final numeric answer
                answer = extract_final_answer(cot)
                if not answer:
                    continue
                
                # Create reference info (the reasoning/solution part)
                ref_info = cot[:400] if len(cot) > 400 else cot
                
                # Format for SFT - teach the model to read and cite the reference
                sft_entry = {
                    "input": f"[参照情報: {ref_info}]\n\nUser: {question}\nModel:",
                    "output": f" 参照情報によると、答えは {answer} です。",
                    "question": question,
                    "answer": answer
                }
                samples.append(sft_entry)
                
            except Exception as e:
                continue
    
    print(f"Generated {len(samples)} SFT samples")
    
    with open(OUTPUT_FILE, 'w') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    print(f"Saved to {OUTPUT_FILE}")
    
    # Show example
    if samples:
        print("\n--- Example ---")
        print(f"Input: {samples[0]['input'][:300]}...")
        print(f"Output: {samples[0]['output']}")

if __name__ == "__main__":
    main()
