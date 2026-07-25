import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import our custom components
from core.patcher import patch_model
from engine.speculative_runner import SpeculativeEngine

class Config:
    def __init__(self, device="mps", max_seq_len=2048, speculative_k=2):
        self.device = device
        self.max_seq_len = max_seq_len
        self.speculative_k = speculative_k

def sync_device(device: str):
    """Safely synchronizes streams depending on the active hardware backend."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def run_baseline_naive(model_id, prompt, max_new_tokens, device):
    """
    Runs standard autoregressive generation using your custom patched attention 
    function without a KV cache (or with it explicitly disabled).
    """
    print(f"\n[Baseline] Loading model for custom patched naive run: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to(device)
    
    # Apply your custom monkey-patched attention (custom_cache remains None by default)
    patch_model(model)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    
    print("[Baseline] Running custom patched generation loop...")
    sync_device(device)
    t0 = time.perf_counter()
    
    current_ids = input_ids
    generated_tokens = []
    
    # Manual token-by-token loop using the custom attention patch without KV cache
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=current_ids, use_cache=False)
            logits = outputs.logits[:, -1, :]
            
            # Greedy next token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = next_token.item()
            
            generated_tokens.append(token_id)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            
            if token_id == tokenizer.eos_token_id:
                break
                
    sync_device(device)
    decode_time = time.perf_counter() - t0
    
    total_tokens = len(generated_tokens)
    throughput = total_tokens / decode_time if decode_time > 0 else 0
    
    print("\n----------------------------------------")
    print("BASELINE PATCHED NAIVE RUN METRICS:")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Decode Time:            {decode_time:.2f} s")
    print(f"Throughput:             {throughput:.2f} tok/sec")
    print("----------------------------------------")
    
    # Cleanup memory
    del model
    # torch.cuda.empty_cache() if device == "cuda" else None
    torch.mps.empty_cache() if device == "mps" else None
    
    return throughput

def run_speculative_engine(target_id, draft_id, prompt, max_new_tokens, speculative_k, device):
    """
    Runs our custom Speculative Engine utilizing static KV Caches and custom patching.
    """
    print(f"\n[Speculative Engine] Initializing on {device.upper()}...")
    print(f"Target Model: {target_id}")
    print(f"Draft Model:  {draft_id}")
    
    config = Config(device=device, speculative_k=speculative_k)
    
    tokenizer = AutoTokenizer.from_pretrained(target_id)
    
    target_model = AutoModelForCausalLM.from_pretrained(
        target_id, torch_dtype=torch.float16
    ).to(device)
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_id, torch_dtype=torch.float16
    ).to(device)
    
    engine = SpeculativeEngine(target_model, draft_model, tokenizer, config)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    print("Generating (Speculative Decoding with Greedy Matching)...")
    generated_tokens, stats = engine.generate(
        input_ids, 
        max_new_tokens=max_new_tokens, 
        do_sample=False
    )
    
    full_ids = torch.cat([input_ids, torch.tensor([generated_tokens], device=device)], dim=-1)
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    
    total_tokens = len(generated_tokens)
    throughput = total_tokens / stats["decode_time"] if stats["decode_time"] > 0 else 0
    acceptance_rate = (stats["draft_tokens_accepted"] / stats["draft_tokens_proposed"] * 100) if stats["draft_tokens_proposed"] > 0 else 0
    
    print("\n========================================")
    print("FINAL OUTPUT:")
    print(text)
    print("========================================")
    print("KV Cache + SPECULATIVE DECODING PROFILER")
    print("========================================")
    print(f"Total Tokens Generated: {total_tokens}")
    print(f"Time To First Token:    {stats['ttft']*1000:.2f} ms")
    print(f"Decode Time:            {stats['decode_time']:.2f} s")
    print(f"Throughput:             {throughput:.2f} tok/sec")
    print("----------------------------------------")
    print(f"Draft Tokens Proposed:  {stats['draft_tokens_proposed']}")
    print(f"Draft Tokens Accepted:  {stats['draft_tokens_accepted']}")
    print(f"Acceptance Rate:        {acceptance_rate:.1f}%")
    print("----------------------------------------")
    print(f"Target Forward Passes:  {stats['target_forward_passes']}")
    print(f"Forward Passes Saved:   {stats['draft_tokens_accepted']} (Skipped heavy memory reads)")
    print("========================================")
    
    return throughput

def main():
    parser = argparse.ArgumentParser(description="Nano-Infer CLI: Baseline vs Speculative Decoding Profiler")
    parser.add_argument("--target", type=str, default="Qwen/Qwen2.5-1.5B", help="Hugging Face Target Model ID")
    parser.add_argument("--draft", type=str, default="Qwen/Qwen2.5-0.5B", help="Hugging Face Draft Model ID")
    parser.add_argument("--prompt", type=str, default="The most critical algorithm in digital signal processing for analyzing frequency domains is", help="Input prompt")
    parser.add_argument("--tokens", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--k", type=int, default=2, help="Speculative lookahead window size (k)")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"], help="Compute device")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip the naive baseline run to save time")
    
    args = parser.parse_args()
    
    print("========================================")
    print("       NANO-INFER BENCHMARK CLI         ")
    print("========================================")
    
    baseline_throughput = 0.0
    if not args.skip_baseline:
        baseline_throughput = run_baseline_naive(
            model_id=args.target,
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            device=args.device
        )
    else:
        print("\n[CLI] Skipping baseline naive run as requested.")
        
    spec_throughput = run_speculative_engine(
        target_id=args.target,
        draft_id=args.draft,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        speculative_k=args.k,
        device=args.device
    )
    
    if not args.skip_baseline and baseline_throughput > 0:
        speedup = spec_throughput / baseline_throughput
        print(f"\nSPEEDUP SUMMARY: Speculative decoding throughput is {spec_throughput:.2f} tok/sec vs Baseline {baseline_throughput:.2f} tok/sec ({speedup:.2f}x).")

if __name__ == "__main__":
    main()