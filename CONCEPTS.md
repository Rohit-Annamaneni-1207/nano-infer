## Concept 1: KV Caching as a Stateful Update

### The Mathematical Bottleneck
In standard scaled dot-product attention:
$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

During autoregressive decoding at step $t$, the query $Q_t$ is just a single token vector $Q_t\in\mathbb{R}^{1\times d}$. However, it must attend to all past tokens, meaning $K_{1:t}\in\mathbb{R}^{t\times d}$ and $V_{1:t}\in\mathbb{R}^{t\times d}$. 

Without caching, computing $K_{1:t}$ requires passing the entire sequence $[x_1,\dots,x_t]$ through the linear projection weights again. This scales computation at $O(t^2)$ per generated sequence.

### The KV Cache Solution
Because the projections for $K$ and $V$ depend *only* on the current token and not on future tokens (due to causal masking), $K_{1:t-1}$ and $V_{1:t-1}$ are immutable. We can treat the attention mechanism like a stateful recursive system. We only compute $K_t=x_tW_K$ and $V_t=x_tW_V$, and append them to the cached states. Compute drops to $O(1)$ per step, but memory footprint grows by $O(t)$.

### Systems Design Pattern: Pre-allocated Buffers vs. Dynamic Concatenation
**The Anti-Pattern:** Using `torch.cat([past_key, current_key], dim=-2)` at every step. This forces PyTorch to allocate a new tensor and copy all historical data to new memory addresses, causing severe memory fragmentation and allocator overhead. 

**The Correct Pattern (Static Allocation):**
Allocate a massive zero-tensor at initialization: 
`self.k_cache = torch.zeros((batch_size, num_heads, max_seq_len, head_dim))`
At step $t$, slice and insert: `self.k_cache[:, :, t:t+1, :] = new_k`.
This ensures contiguous memory and zero reallocation overhead.

### Interview FAQ
**Q: Why is LLM Prefill Compute-Bound but Decode is Memory-Bandwidth Bound?**
*   **Prefill:** We process the prompt of length $N$ in parallel. We perform massive matrix multiplications ($N\times d$ multiplied by $d\times d$ weights). The GPU's arithmetic logic units (ALUs) are fully saturated.
*   **Decode:** We process $N=1$. The math is trivial (vector-matrix multiplication), but to do it, we must load the massive model weights and the entire KV cache from High Bandwidth Memory (HBM) to the SRAM registers for *every single token*. The GPU ALUs sit idle waiting for memory to move.