import torch
import torch.nn.functional as F

def top_p_sample(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """
    Filters logits using Top-P (Nucleus) sampling and picks the next token.
    logits shape expected: (batch_size, vocab_size)
    """
    # 1. Apply temperature scaling and upcast to float32 to prevent MPS precision underflow
    logits = (logits / temperature).float()
    
    # 2. Convert raw logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # 3. Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # 4. Calculate the cumulative sum of the sorted probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 5. Create a mask to remove tokens where the cumulative sum exceeds top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Shift the mask one spot to the right. 
    # This guarantees we always keep at least the single most probable token.
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 6. Scatter the mask back to the original vocabulary indices
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    
    # 7. Set the logits of removed tokens to negative infinity
    logits[indices_to_remove] = float('-inf')
    
    # 8. Re-normalize the surviving tokens into a new probability distribution
    filtered_probs = F.softmax(logits, dim=-1)
    
    # 9. Roll the weighted dice (CPU offload bypass for MPS stability)
    next_token = torch.multinomial(filtered_probs.cpu(), num_samples=1).to(logits.device)
    
    return next_token