# nano-infer

Nano-Infer is a small inference benchmarking sandbox for causal language models. The repository focuses on the mechanics of prefill and decode optimization: static KV caching, monkey-patched attention, speculative decoding, and lightweight profiling of time-to-first-token and decode throughput.

## What Is Implemented

The codebase is centered around Hugging Face models and a custom attention path that injects its own cache management:

- `config/settings.py` defines the structured runtime configuration for model, cache, and generation settings.
- `core/kv_cache.py` implements a preallocated key/value cache with in-place updates and cheap pointer resets.
- `core/patcher.py` monkey-patches a model’s attention layers so they use the custom cache, rotary position embeddings, grouped-query head expansion, and causal masking.
- `core/sampler.py` provides top-p sampling for speculative generation.
- `engine/runner.py` runs a single-model generation path with the patched attention and static cache.
- `engine/speculative_runner.py` runs the target/draft speculative decoding loop, including draft generation, verification, cache alignment, and acceptance tracking.
- `engine/profiler.py` prints inference telemetry such as TTFT, decode speed, throughput, and speculative acceptance rate.
- `main_cli.py` is the benchmark entrypoint that compares a patched baseline run against speculative decoding.

## How It Works

The design follows the usual LLM inference split:

- Prefill processes the prompt in parallel and fills the cache.
- Decode generates one token at a time, reusing cached keys and values instead of recomputing the prompt.

The cache is static: it allocates the maximum sequence length up front and writes new K/V states into the existing buffer. Attention then reads the full cached prefix from that buffer, so the model still attends to the entire history even though each update only appends the current token’s states.

Grouped-query attention is handled by repeating the cached K/V heads to match the number of query heads before the attention matmul.

## Benchmark CLI

The main entrypoint is `main_cli.py`.

```bash
python main_cli.py --tokens 100 --k 4
```

Useful flags:

- `--target`: Hugging Face target model ID for the main model.
- `--draft`: Hugging Face draft model ID used by speculative decoding.
- `--prompt`: input prompt.
- `--tokens`: number of new tokens to generate.
- `--k`: speculative lookahead window.
- `--device`: `mps`, `cuda`, or `cpu`.
- `--skip-baseline`: skip the naive patched baseline run.

The CLI prints:

- baseline throughput for the patched autoregressive loop
- speculative decoding throughput
- TTFT
- decode time
- draft tokens proposed and accepted
- acceptance rate
- target forward passes saved

## Configuration

Runtime settings live in `config/settings.py` and are exposed as a global `settings` object. The configuration is split into:

- model settings: model ID, device, and dtype
- cache settings: max batch size and max sequence length
- generation settings: max new tokens, temperature, top-p, and top-k

The model dtype is stored as a readable string in config and converted to a `torch.dtype` via the `torch_dtype` helper property.

## Repo Notes

- `run_benchmark.py` is currently empty and appears to be reserved for future benchmark orchestration.
- The `dashboard/` and `tests/` packages are present as placeholders, but their current files are empty.
- This README intentionally describes the implemented code paths in the repository and leaves out any speculative main entrypoint outside the current surface.

## Core Concepts

The implementation is built around a few systems ideas:

- KV caching turns decode into a stateful update instead of recomputing all past tokens.
- Static buffers avoid repeated tensor reallocation and memory fragmentation.
- RoPE is applied inside the patched attention path so position information is preserved without adding positional embeddings.
- Speculative decoding reduces heavy target-model forward passes by drafting multiple tokens with a smaller model and verifying them with the target model.

## Sample run
```bash
python main_cli.py --tokens 100 --k 4
```

```bash
========================================
       NANO-INFER BENCHMARK CLI         
========================================

[Baseline] Loading model for custom patched naive run: Qwen/Qwen2.5-1.5B...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:00<00:00, 2339.96it/s]
Successfully monkey-patched 28 attention layers.
[Baseline] Running custom patched generation loop...

----------------------------------------
BASELINE PATCHED NAIVE RUN METRICS:
Total Tokens Generated: 100
Decode Time:            5.92 s
Throughput:             16.89 tok/sec
----------------------------------------

[Speculative Engine] Initializing on MPS...
Target Model: Qwen/Qwen2.5-1.5B
Draft Model:  Qwen/Qwen2.5-0.5B
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:00<00:00, 1591.94it/s]
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 1242.96it/s]
Successfully monkey-patched 28 attention layers.
Successfully monkey-patched 24 attention layers.
Generating (Speculative Decoding with Greedy Matching)...

========================================
FINAL OUTPUT:
The most critical algorithm in digital signal processing for analyzing frequency domains is!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
========================================
KV Cache + SPECULATIVE DECODING PROFILER
========================================
Total Tokens Generated: 102
Time To First Token:    72.61 ms
Decode Time:            3.41 s
Throughput:             29.89 tok/sec
----------------------------------------
Draft Tokens Proposed:  120
Draft Tokens Accepted:  72
Acceptance Rate:        60.0%
----------------------------------------
Target Forward Passes:  30
Forward Passes Saved:   72 (Skipped heavy memory reads)
========================================

SPEEDUP SUMMARY: Speculative decoding throughput is 29.89 tok/sec vs Baseline 16.89 tok/sec (1.77x).
```
