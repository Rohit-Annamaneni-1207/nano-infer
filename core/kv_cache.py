import torch

class KVCache:
    def __init__(
        self, 
        batch_size: int, 
        max_seq_len: int, 
        num_kv_heads: int, 
        head_dim: int, 
        dtype: torch.dtype = torch.float16, 
        device: str = "cuda"
    ):
        """
        Pre-allocates the Key and Value cache tensors for a single transformer layer.
        """
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # Static allocation: (Batch, Heads, Sequence, Head_Dim)
        # Using zeros to ensure clean initialization, though empty() is slightly faster.
        self.k_cache = torch.zeros(
            (batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device
        )
        self.v_cache = torch.zeros(
            (batch_size, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device
        )
        
        # Explicit position tracking
        self.current_pos = 0

    def update(self, k_states: torch.Tensor, v_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inserts new KV states into the cache and returns the full historical context.
        
        Args:
            k_states: Tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            v_states: Tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            
        Returns:
            Tuple of (k_cache, v_cache) sliced up to the current total sequence length.
        """
        seq_len = k_states.size(2)
        
        if self.current_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"KV Cache overflow: attempting to insert {seq_len} tokens "
                f"at position {self.current_pos}, but max_seq_len is {self.max_seq_len}."
            )
            
        # In-place slice assignment to avoid reallocation
        self.k_cache[:, :, self.current_pos : self.current_pos + seq_len, :] = k_states
        self.v_cache[:, :, self.current_pos : self.current_pos + seq_len, :] = v_states
        
        self.current_pos += seq_len
        
        # Return a view of the cache up to the current position
        return (
            self.k_cache[:, :, :self.current_pos, :],
            self.v_cache[:, :, :self.current_pos, :]
        )

    def get_seq_len(self) -> int:
        return self.current_pos

    def reset(self):
        """
        Resets the position pointer without freeing memory. 
        Old data will simply be overwritten on the next update.
        """
        self.current_pos = 0