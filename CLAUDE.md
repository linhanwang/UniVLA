# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UniVLA is a unified Vision-Language-Action model for robotics and autonomous driving. It extends Emu3 (a multimodal LLM) with action prediction capabilities using a Mixture-of-Experts (MoE) architecture. The model tokenizes vision, language, and actions into a shared token space, enabling interleaved video-action training.

## Setup

```shell
uv sync
```

Uses uv for environment management. PyTorch CUDA 13.0 builds are pulled from the PyTorch index configured in `pyproject.toml`. Flash attention is provided by PyTorch's built-in SDPA — no separate `flash-attn` package needed.

### Unified H200 / L40s env

A single env (torch 2.10.0 + cu130, no `flash-attn` package) runs on both H200 (Hopper sm_90) and L40s (Ada sm_89). It works because `Emu3SdpaAttention` in `emu3/mllm/modeling_emu3.py` calls SDPA with `attn_mask=None, is_causal=True`, and `Emu3Model` skips building a 4D causal mask on the FA2/SDPA paths. With no explicit mask, PyTorch SDPA dispatches to its fast backend per arch:

- L40s (Ada) → FA2 backend (`SDPBackend.FLASH_ATTENTION`).
- H200 (Hopper) → cuDNN flash backend (`SDPBackend.CUDNN_ATTENTION`).

This is safe because inputs are right-padded and pad positions carry `label=-100`: causal attention alone keeps non-pad outputs from attending to pads, and pad outputs don't contribute to loss. Do **not** reintroduce an explicit `attention_mask` into the FA2/SDPA paths — passing one knocks SDPA off the fast backends (silently regresses H200 perf). If padding semantics ever change (left-padding, or labels not -100 on pads), this assumption needs re-examining. See commit df6e509.

## Training Commands

All training uses `torchrun` with DeepSpeed ZeRO-3. Set `PYTHONPATH=$(pwd)` before running.

```shell
# World model pretraining (video generation)
bash scripts/pretrain/train_video_1node.sh

# Policy learning (simulator benchmarks)
bash scripts/simulator/calvin/train_calvin_abcd_video.sh
bash scripts/simulator/libero/train_libero_video.sh
bash scripts/simulator/simplerenv/train_simplerenv_bridge_video.sh

# Real robot
bash scripts/real_aloha/train_real_aloha_chunknorm.sh
```

Training entry point is `train/train_moe.py`, which uses HuggingFace `Trainer` with custom argument dataclasses (`ModelArguments`, `DataArguments`, `TrainingArguments`).

## Architecture

### Model hierarchy

- **Emu3MoE** (`emu3/mllm/modeling_emu3.py`): Core model extending Emu3ForCausalLM with MoE action experts. Config in `emu3/mllm/configuration_emu3.py` defines both the main LLM config and `action_config` for action expert layers.
- **Emu3Tokenizer / Emu3Processor** (`emu3/mllm/`): Text tokenizer and multimodal processor that handles vision token encoding via a VQ model (Emu3-VisionTokenizer).

### Action tokenization (two approaches)

1. **OpenVLA-style discrete bins** (`models/tokenizer/action_tokenizer.py`): Uniform binning of continuous actions into N bins mapped to the tail of the vocabulary. Used when `actions_format="openvla"`.
2. **FAST (DCT + BPE)** (`pretrain/fast/processing_action_tokenizer.py`): `UniversalActionProcessor` applies DCT to action chunks, quantizes coefficients, then uses a BPE tokenizer. Used when `actions_format="fast"`. Pretrained tokenizers live in `pretrain/fast*/` directories.

### Data pipeline

- `train/datasets.py`: Dataset classes (`Emu3SFTDataset`, `Emu3WorldModelDataset`, `Emu3RealRobotDataset`, `Emu3CoTDataset`) that load preprocessed pickle files containing VQ-encoded image codes and action sequences.
- `models/tokenizer/emu3_tokenizer.py`: Vision encoding pipeline and data preprocessing utilities. Run standalone to encode raw images into VQ codes for different datasets.
- `train/dataset/normalize_pi0.py`: Action normalization utilities.

### Data format

Training data is stored as pickle files containing lists of dicts with keys like `text`, `image` (paths to `.npy` VQ codes), `action`, and optionally `gripper_image`. Normalizer configs are in `configs/normalizer_*/norm_stats.json`.

### Inference

- `models/inference/inference_action.py`: Action prediction inference with constrained decoding (restricts generation to action token IDs).
- `models/inference/inference_vision.py`: Vision/video generation inference, supports vLLM acceleration.

### Key training flags

- `--apply_loss_on_only_vision True`: World model pretraining (predict next video frame).
- `--apply_loss_on_only_action True`: Policy learning (predict actions only).
- `--video_format "interleave"`: Interleaved vision-action MDP training.
- `--actions_format "fast"`: Use FAST action tokenizer (vs `"openvla"` for discrete bins).
- `--post_training True`: World model post-training mode (uses `Emu3WorldModelDataset`).

### Policy head

`models/policy_head/noise_schedulers.py`: FlowMatchingScheduler for diffusion-based action denoising (Beta or Uniform timestep sampling).

## Reference Code

- `emu3/`: Base Emu3 model code (from BAAI), moved to repo root. The core model, tokenizer, and processor classes are defined here.
- `reference/RoboVLMs/`: Evaluation code for SimplerEnv benchmark.

## Benchmark-specific docs

See `docs/calvin.md`, `docs/libero.md`, `docs/simplerenv.md`, `docs/aloha.md` for benchmark setup and evaluation instructions.
