import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.patcher import patch_model
from core.kv_cache import KVCache
from engine.profiler import InferenceProfiler
from config.settings import settings

class NanoInferEngine:
    def __init__(self):
        # Everything is now driven by Pydantic
        self.device = settings.model.device
        print(f"Loading model {settings.model.model_id} to {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model.model_id)
        hf_model = AutoModelForCausalLM.from_pretrained(
            settings.model.model_id, 
            torch_dtype=settings.model.torch_dtype
        ).to(self.device)
        
        self.model = patch_model(hf_model)
        self.config = self.model.config

    def _init_caches(self):
        """Initializes caches using strict limits from CacheConfig."""
        for layer in self.model.model.layers:
            cache = KVCache(
                batch_size=settings.cache.max_batch_size,
                max_seq_len=settings.cache.max_seq_len,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.hidden_size // self.config.num_attention_heads,
                dtype=settings.model.torch_dtype,
                device=self.device
            )
            layer.self_attn.custom_cache = cache

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        profiler = InferenceProfiler(device=self.device)
        profiler.capture_memory("baseline_loaded")

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]
        
        # We only allocate the cache once up to the configured max_seq_len
        self._init_caches()
        
        max_gen = settings.generation.max_new_tokens
        print(f"\nStarting Generation (Prompt: {prompt_len} tokens, Max Gen: {max_gen} tokens)")
        
        # ==========================================
        # PHASE 1: PREFILL
        # ==========================================
        profiler.start_timer()
        position_ids = torch.arange(0, prompt_len, dtype=torch.long, device=self.device).unsqueeze(0)
        
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False 
        )
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = [next_token_id.item()]
        
        ttft_ms = profiler.stop_timer()
        profiler.capture_memory("post_prefill")

        # ==========================================
        # PHASE 2: DECODE
        # ==========================================
        profiler.start_timer()
        current_input = next_token_id
        current_pos = prompt_len
        
        for i in range(max_gen - 1):
            position_ids = torch.tensor([[current_pos]], dtype=torch.long, device=self.device)
            
            outputs = self.model(
                input_ids=current_input,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
                
            generated_ids.append(next_token_id.item())
            current_input = next_token_id
            current_pos += 1
            
        total_decode_ms = profiler.stop_timer()
        profiler.capture_memory("post_decode")
        
        profiler.print_report(prompt_len, len(generated_ids), ttft_ms, total_decode_ms)
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


if __name__ == "__main__":
    engine = NanoInferEngine()
    prompt = (
        "Explain the fundamental difference between compute-bound operations "
        "and memory-bandwidth bound operations in the context of hardware accelerators."
    )
    # Notice we no longer pass max_new_tokens here!
    response = engine.generate(prompt)
    print(f"Response:\n{response}")