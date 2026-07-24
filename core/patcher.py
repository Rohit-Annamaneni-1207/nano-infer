import math
import torch
import torch.nn.functional as F
import types
from typing import Optional, Any
from .kv_cache import KVCache

def custom_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Any] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    # 1. Linear Projections & Immediate Standardization
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # 2. Rotary Positional Embeddings (RoPE)
    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        cos, sin = self.rotary_emb(value_states, position_ids)
    
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 3. Inject Our Bound KV Cache
    if hasattr(self, "custom_cache") and self.custom_cache is not None:
        past_len = self.custom_cache.current_pos
        key_states, value_states = self.custom_cache.update(key_states, value_states)
    else:
        past_len = 0

    # 4. GQA Repeat
    if self.num_key_value_heads < self.num_heads:
        num_kv_groups = self.num_heads // self.num_key_value_heads
        key_states = key_states.repeat_interleave(num_kv_groups, dim=1)
        value_states = value_states.repeat_interleave(num_kv_groups, dim=1)

    # 5. Scaled Dot-Product Attention
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # --- CUSTOM CAUSAL MASK ---
    if q_len > 1:
        mask = torch.full((q_len, q_len), float('-inf'), device=query_states.device)
        mask = torch.triu(mask, diagonal=1)
        if past_len > 0:
            past_mask = torch.zeros((q_len, past_len), device=query_states.device)
            mask = torch.cat([past_mask, mask], dim=1)
        
        mask = mask[None, None, :, :]
        attn_weights = attn_weights + mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # 6. Transpose back for output projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    # 7. Safe Return Tuple Matching Qwen2DecoderLayer Expectation
    # We must return exactly 2 items to satisfy: hidden_states, _ = self.self_attn(...)
    return (attn_output, None)


def patch_model(model):
    patched_count = 0
    for layer in model.model.layers:
        layer.self_attn.num_heads = model.config.num_attention_heads
        layer.self_attn.num_key_value_heads = model.config.num_key_value_heads
        layer.self_attn.head_dim = model.config.hidden_size // model.config.num_attention_heads
        layer.self_attn.hidden_size = model.config.hidden_size
        layer.self_attn.custom_cache = None  
        
        layer.self_attn.forward = types.MethodType(custom_attention_forward, layer.self_attn)
        patched_count += 1
        
    print(f"Successfully monkey-patched {patched_count} attention layers.")
    return model