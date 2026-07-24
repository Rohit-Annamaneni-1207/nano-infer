import torch
import time
from typing import Dict

class InferenceProfiler:
    def __init__(self, device: str = "mps"):
        self.device = device
        self.metrics = {}
        
        # Determine the correct synchronization function based on backend
        if self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.sync = torch.mps.synchronize
            self.get_mem = lambda: torch.mps.current_allocated_memory() / (1024 ** 2)  # MB
        elif self.device == "cuda" and torch.cuda.is_available():
            self.sync = torch.cuda.synchronize
            self.get_mem = lambda: torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            # Fallback for CPU
            self.sync = lambda: None
            self.get_mem = lambda: 0.0

    def start_timer(self):
        """Forces the GPU to finish pending work, then starts the clock."""
        self.sync()
        self.t0 = time.perf_counter()

    def stop_timer(self) -> float:
        """Forces the GPU to finish the measured work, then stops the clock."""
        self.sync()
        return (time.perf_counter() - self.t0) * 1000  # Return in milliseconds

    def capture_memory(self, phase_name: str):
        """Records the current VRAM usage."""
        mem_mb = self.get_mem()
        self.metrics[f"{phase_name}_memory_mb"] = mem_mb
        return mem_mb

    def print_report(self, prompt_tokens: int, generated_tokens: int, ttft_ms: float, total_decode_ms: float):
        """Prints a clean benchmark summary."""
        decode_speed = (generated_tokens / (total_decode_ms / 1000)) if total_decode_ms > 0 else 0
        
        print("\n" + "="*40)
        print("🚀 INFERENCE PROFILING REPORT")
        print("="*40)
        print(f"Hardware Backend : {self.device.upper()}")
        print(f"Context          : {prompt_tokens} prompt -> {generated_tokens} generated")
        print("-" * 40)
        print(f"Time To First Token (TTFT) : {ttft_ms:.2f} ms")
        print(f"Decode Speed               : {decode_speed:.2f} tokens/sec")
        print("-" * 40)
        for phase, mem in self.metrics.items():
            print(f"Peak VRAM ({phase})   : {mem:.2f} MB")
        print("="*40 + "\n")