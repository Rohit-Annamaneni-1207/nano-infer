import torch
from pydantic import BaseModel, Field
from typing import Literal, Optional

class ModelConfig(BaseModel):
    """Configuration for model loading and hardware mapping."""
    model_id: str = Field(
        default="Qwen/Qwen2.5-0.5B", 
        description="The HuggingFace hub ID or local path to the model."
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="mps", 
        description="Hardware backend for inference."
    )
    dtype: Literal["float16", "bfloat16", "float32"] = Field(
        default="float16", 
        description="Precision format for the model weights."
    )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Helper property to convert the string dtype to a torch dtype."""
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        return mapping[self.dtype]


class CacheConfig(BaseModel):
    """Constraints for the KV Cache allocations."""
    max_batch_size: int = Field(
        default=1, 
        ge=1, 
        description="Maximum concurrent sequences."
    )
    max_seq_len: int = Field(
        default=2048, 
        ge=1, 
        description="Maximum total tokens (prompt + generated) allowed in the cache."
    )


class GenerationConfig(BaseModel):
    """Hyperparameters for the autoregressive decoding loop."""
    max_new_tokens: int = Field(
        default=128, 
        ge=1, 
        le=8192,
        description="Maximum number of tokens to generate."
    )
    temperature: float = Field(
        default=0.7, 
        ge=0.0, 
        le=2.0,
        description="Controls randomness. 0.0 is greedy decoding."
    )
    top_p: float = Field(
        default=0.9, 
        ge=0.0, 
        le=1.0,
        description="Nucleus sampling threshold."
    )
    top_k: int = Field(
        default=50, 
        ge=0,
        description="Limits sampling to the K most likely tokens."
    )


class EngineSettings(BaseModel):
    """Master configuration object for NanoInfer."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

# Global singleton for easy import across the project
settings = EngineSettings()