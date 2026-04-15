# Training Speedup Ideas

Context: Emu3MoE (LLaMA-family: RMSNorm + RoPE + SwiGLU + GQA/MHA) training on H200 (140 GB). Current baseline uses ~103 GB at bs=16, ga=2, GC on, `torch.compile` with `reduce-overhead`, DeepSpeed ZeRO-2. GPU util is 100%, so dataloader is not the bottleneck.

Ranked by likely payoff vs. engineering effort.

## 1. Fused cross-entropy (biggest single win)

Emu3's vocab is ~184k (text + vision tokens), plus action vocab. The naive CE path materializes `logits = lm_head(hidden)` with shape `[B, T, V]` = `[16, 2400, ~190k]` ≈ **55 GB** in BF16. This is the dominant activation.

- **Investigate**: grep `modeling_emu3.py` for `lm_head(`, `cross_entropy`, `CrossEntropyLoss`. Confirm logits are materialized before the loss.
- **Fix**: [Liger Kernel](https://github.com/linkedin/Liger-Kernel)'s `LigerFusedLinearCrossEntropy` computes CE directly from `hidden @ lm_head.weight` in chunks without materializing full logits.
- **Expected gain**: 30-50 GB activation memory freed, 15-25% wall-clock speedup on large-vocab LLMs. May enable bs=32 without LoRA.

## 2. Liger kernels for the rest of the stack

Fused RMSNorm, RoPE, SwiGLU in addition to CE. Typical combined speedup on LLaMA is 15-25%, memory drops 20-40%.

- **Caveat**: Emu3 is a fork, so `apply_liger_kernel_to_llama()` will not monkey-patch correctly. Import individual kernels and swap them into `Emu3DecoderLayer` / `Emu3RMSNorm` manually.
- **Effort**: tens of lines, surgical.

## 3. Attention backend

Check what `Emu3Attention` actually calls in `reference/Emu3/emu3/mllm/modeling_emu3.py`.

- **cuDNN SDPA backend** (Hopper-native, fast for BF16): force with `torch.backends.cuda.sdp_kernel(enable_cudnn=True)`.
- **FlashAttention-3** (Hopper-specific, ~1.5-2× FA2): requires the `flash-attn` v3 build.
- **FlexAttention** (PyTorch 2.5+): great for custom masks, see next item.

If the current code path is eager attention or an old FA1/FA2 call, upgrading is easy speedup.

## 4. FlexAttention for interleaved video-action masking

`--video_format interleave` means a structured attention mask. If the current code builds a full `[T, T]` mask tensor, FlexAttention's block-sparse support can skip masked regions entirely. For seq 2400 this could be substantial if the mask is mostly causal within segments.

## 5. Optimizer

- **Fused AdamW**: if `torch.optim.AdamW` is used without `fused=True`, flip the flag. One-line change.
- **8-bit optimizer**: `bitsandbytes.AdamW8bit` or `PagedAdamW8bit` saves ~15-20 GB of optimizer state. Frees budget for bigger batch.

## 6. Compile scope

`torch.compile(model)` on the whole HF model often leaves graph breaks at the loss / HF output-dict construction. Compiling only `Emu3DecoderLayer` per-block sometimes compiles more aggressively and avoids recompiles.

- **Investigate**: run with `TORCH_LOGS="recompiles,graph_breaks"` and see where breaks happen.
- **Try**: `Emu3DecoderLayer.forward = torch.compile(Emu3DecoderLayer.forward, mode="max-autotune")` per-layer.

## 7. Partial gradient checkpointing

Current GC is all-or-nothing via HF flag.

- **Strategy**: checkpoint every-N layers, or skip action-expert layers (smaller path).
- **Expected gain**: ~12-15% wall-clock if half the layers are un-checkpointed.
- **Caveat**: `torch.compile` + selective checkpointing can have graph breaks at checkpoint boundaries. Verify.
- **Caveat**: 103 GB peak is likely during optimizer step (ZeRO-2 all-reduce buffers). Partial GC grows forward-pass peak, so watch both.

## 8. FP8 via torchao

H200 is Hopper → native FP8 tensor cores.

- **API**: `convert_to_float8_training(model, config, module_filter_fn=...)`. Filter out `lm_head`, embeddings, anything < 1024 dim.
- **Expected gain**: 1.3-1.5× wall-clock on LLM training in practice.
- **Risks**:
  - DeepSpeed ZeRO-2 + FP8 is not torchao's blessed path (FSDP2 is). May hit all-reduce dtype or amax-sync issues.
  - `torch_compile` + `reduce-overhead` (CUDA graphs) conflicts with FP8 amax history updates. Likely need to drop to `mode="default"` or disable compile.
  - First ~500-2000 steps with dynamic scaling can be unstable; delayed scaling is more stable but more intrusive.
- **Effort estimate**: half a day (happy path) to a week (unhappy path). Not free.

## 9. FSDP2 vs DeepSpeed ZeRO-2

FSDP2 (per-parameter sharding, PyTorch 2.4+) plays much better with `torch.compile` and with torchao FP8 than DeepSpeed does.

- **Effort**: migration is nontrivial.
- **Payoff**: unlocks the modern stack (FP8 + compile + selective GC) cleanly.

## Where to start

1. Grep `lm_head`, `cross_entropy`, attention forward. If CE is unfused on ~184k vocab, that's the highest-leverage, well-understood fix.
2. Confirm attention backend. Easy upgrade if on eager/FA1.
3. Everything else is incremental.

## Verifying the baseline first

Before optimizing:

- Run with `TORCH_LOGS="recompiles"` to confirm `torch.compile` isn't silently recompiling every step.
- Check per-step time variance in logs — flat = compute-bound, spiky = something stalls.
- `nvidia-smi dmon -s u` for SM-active %. 100% "GPU util" in nvtop just means a kernel was scheduled; SM-active tells you tensor cores are actually busy. <70% SM-active with 100% GPU util means memory-BW or launch-overhead bound, not compute-bound — FP8 won't help as much.
