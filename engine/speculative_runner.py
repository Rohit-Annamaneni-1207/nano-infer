import torch
import time
from typing import List, Tuple
from core.kv_cache import KVCache
from core.patcher import patch_model
from core.sampler import top_p_sample

class SpeculativeEngine:
    def __init__(self, target_model, draft_model, tokenizer, config):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.device = config.device
        self.k = config.speculative_k 
        
        patch_model(self.target_model)
        patch_model(self.draft_model)
        
        for layer in self.target_model.model.layers:
            layer.self_attn.custom_cache = KVCache(
                batch_size=1, max_seq_len=config.max_seq_len, 
                num_kv_heads=target_model.config.num_key_value_heads,
                head_dim=target_model.config.hidden_size // target_model.config.num_attention_heads,
                device=self.device
            )
            
        for layer in self.draft_model.model.layers:
            layer.self_attn.custom_cache = KVCache(
                batch_size=1, max_seq_len=config.max_seq_len,
                num_kv_heads=draft_model.config.num_key_value_heads,
                head_dim=draft_model.config.hidden_size // draft_model.config.num_attention_heads,
                device=self.device
            )

    @torch.no_grad()
    def _generate_draft(self, input_ids: torch.Tensor, start_pos: int, top_p: float, temperature: float, do_sample: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        draft_tokens = []
        draft_probs = [] 
        current_input = input_ids[:, -1:] 
        
        for i in range(self.k):
            pos = torch.tensor([[start_pos + i]], dtype=torch.long, device=self.device)
            outputs = self.draft_model(input_ids=current_input, position_ids=pos, use_cache=False)
            
            logits = outputs.logits[:, -1, :]
            
            if do_sample:
                q_dist = torch.softmax((logits / temperature).float(), dim=-1)
                draft_probs.append(q_dist)
                next_token = top_p_sample(logits, top_p=top_p, temperature=temperature)
            else:
                # Greedy: One-hot distribution and argmax
                q_dist = torch.zeros_like(logits, dtype=torch.float32)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                q_dist.scatter_(1, next_token, 1.0)
                draft_probs.append(q_dist)
                
            draft_tokens.append(next_token.item())
            current_input = next_token
            
        return torch.tensor([draft_tokens], device=self.device), torch.cat(draft_probs, dim=0)

    @torch.no_grad()
    def _verify_draft(self, past_ids: torch.Tensor, draft_ids: torch.Tensor, start_pos: int, 
                      draft_probs: torch.Tensor, top_p: float, temperature: float, do_sample: bool) -> Tuple[List[int], int]:
        verify_input = torch.cat([past_ids[:, -1:], draft_ids], dim=-1)
        seq_len = verify_input.shape[1]
        
        pos = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        outputs = self.target_model(input_ids=verify_input, position_ids=pos, use_cache=False)
        
        target_logits = outputs.logits
        
        if do_sample:
            p_dist = torch.softmax((target_logits / temperature).float(), dim=-1)
            p_dist = torch.nan_to_num(p_dist, nan=0.0, posinf=1.0, neginf=0.0)
        
        accepted_tokens = []
        for i in range(self.k):
            token_id = draft_ids[0, i].item()
            
            if do_sample:
                p_val = p_dist[0, i, token_id]
                q_val = draft_probs[i, token_id]
                
                r = (p_val / q_val).item() if q_val.item() > 0 else 0.0
                
                if torch.rand(1, device=self.device).item() < r:
                    accepted_tokens.append(token_id)
                else:
                    adjusted_dist = torch.clamp(p_dist[0, i] - draft_probs[i], min=0.0)
                    dist_sum = adjusted_dist.sum()
                    
                    if dist_sum.item() > 1e-5:
                        adjusted_dist = (adjusted_dist / dist_sum).cpu()
                        adjusted_dist = torch.nan_to_num(adjusted_dist, nan=0.0)
                        if adjusted_dist.sum() <= 0:
                            adjusted_dist = torch.ones_like(adjusted_dist) / adjusted_dist.numel()
                        correction_token = torch.multinomial(adjusted_dist, 1).item()
                    else:
                        safe_p = p_dist[0, i].cpu()
                        safe_p = torch.nan_to_num(safe_p, nan=0.0)
                        if safe_p.sum().item() > 0:
                            safe_p = safe_p / safe_p.sum()
                        else:
                            safe_p = torch.ones_like(safe_p) / safe_p.numel()
                        correction_token = torch.multinomial(safe_p, 1).item()
                        
                    accepted_tokens.append(correction_token)
                    break
            else:
                # Pure Greedy Strict Equality Check
                target_pred = torch.argmax(target_logits[0, i], dim=-1).item()
                if target_pred == token_id:
                    accepted_tokens.append(token_id)
                else:
                    accepted_tokens.append(target_pred)
                    break
                
        if len(accepted_tokens) == self.k:
            bonus_logits = target_logits[:, self.k, :]
            if do_sample:
                bonus_token = top_p_sample(bonus_logits, top_p=top_p, temperature=temperature)
            else:
                bonus_token = torch.argmax(bonus_logits, dim=-1, keepdim=True)
            accepted_tokens.append(bonus_token.item())
            
        return accepted_tokens, len(accepted_tokens)

    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: torch.Tensor, 
        max_new_tokens: int, 
        do_sample: bool = True,  # <--- Toggle between Greedy and Top-P
        top_p: float = 0.9, 
        temperature: float = 0.7
    ) -> tuple[list[int], dict]:
        t0 = time.perf_counter()
        
        if prompt_ids.shape[1] > 1:
            prefill_ids = prompt_ids[:, :-1]
            seq_len = prefill_ids.shape[1]
            pos = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
            
            self.target_model(input_ids=prefill_ids, position_ids=pos, use_cache=False)
            self.draft_model(input_ids=prefill_ids, position_ids=pos, use_cache=False)
            
        ttft = time.perf_counter() - t0
        current_ids = prompt_ids
        generated_tokens = []
        
        stats = {
            "ttft": ttft,
            "target_forward_passes": 0,
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "decode_time": 0.0
        }
        
        decode_start = time.perf_counter()
        
        while len(generated_tokens) < max_new_tokens:
            valid_memory_length = self.target_model.model.layers[0].self_attn.custom_cache.current_pos
            
            # --- PHASE 1: DRAFT ---
            draft_ids, draft_probs = self._generate_draft(
                current_ids, valid_memory_length, top_p, temperature, do_sample
            )
            stats["draft_tokens_proposed"] += self.k
            
            # --- PHASE 2: VERIFY ---
            accepted_tokens_list, num_accepted = self._verify_draft(
                current_ids, draft_ids, valid_memory_length, draft_probs, top_p, temperature, do_sample
            )
            stats["target_forward_passes"] += 1
            
            m = num_accepted - 1 if num_accepted > 0 else 0
            stats["draft_tokens_accepted"] += m
            generated_tokens.extend(accepted_tokens_list)
            
            # --- PHASE 3: CACHE ALIGNMENT ---
            new_valid_length = valid_memory_length + num_accepted
            for layer in self.target_model.model.layers:
                layer.self_attn.custom_cache.set_position(new_valid_length)
                
            if m > 0:
                for layer in self.draft_model.model.layers:
                    layer.self_attn.custom_cache.set_position(valid_memory_length + m)
                
                missed_token = draft_ids[:, m-1:m]
                pos = torch.tensor([[valid_memory_length + m]], dtype=torch.long, device=self.device)
                self.draft_model(input_ids=missed_token, position_ids=pos, use_cache=False)
            else:
                for layer in self.draft_model.model.layers:
                    layer.self_attn.custom_cache.set_position(valid_memory_length + 1)
                
            last_token = torch.tensor([[accepted_tokens_list[-1]]], device=self.device)
            current_ids = torch.cat([current_ids, last_token], dim=-1)
            
            if accepted_tokens_list[-1] == self.tokenizer.eos_token_id:
                break
                
        stats["decode_time"] = time.perf_counter() - decode_start
        return generated_tokens, stats