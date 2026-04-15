# SIMPLERENV Benchmark

[SIMPLERENV](https://github.com/simpler-env/SimplerEnv) is a benchmark for real-to-sim robot evaluation. 

![](imgs/simpler.png)

## Environment Setup
We follow the [RoboVLMs](https://github.com/Robot-VLAs/RoboVLMs) repository for environment setup. This setup is only for evaluation. The following steps are required to set up the environment:

> Note: when use ray-tracing rendering, please make sure you have the nvoptix.so in /usr/share/nvidia

```shell
# Install dependencies
cd reference/RoboVLMs

# This will install the required environment
bash scripts/setup_simplerenv.sh

# Only for rendering environment.
bash scripts/setup_simplerenv_vla.sh

# Check if the environment is set up correctly
python eval/simplerenv/env_test.py
```

## Dataset Preparation
```shell
# 1. process the dataset (bridge & google)
python tools/process/simplerenv_bridge.py
    --dataset_dir /path/to/bridge_orig/1.0.0 \
    --output_dir /path/to/save/processed_data/bridge

python tools/process/simplerenv_google.py \
  --dataset_dir /path/to/fractal20220817_data \
  --output_dir /path/to/output/simplerenv_google

# 2. extract the vq tokens, need to change the dataset & output path
bash scripts/tokenizer/extract_vq_emu3.sh

# 3. pickle generation for training
python tools/pickle_gen/pickle_generation_simplerenv_bridge.py
```

## Model Training

### FAST Tokenizer
You can fit the FAST tokenizer on the corresponding dataset. Also, you can adjust the scale in tokenizer for more fine-grained tokenization.
```shell
python tools/action_tokenizer/fit_fast.py
```

```shell
bash scripts/simulator/simplerenv/train_simplerenv_bridge_video.sh
```

## Model Evaluation
```shell
cd reference/RoboVLMs

bash scripts/bridge_univla.bash ${CKPT_PATH}

# get results, modify the results path
python eval/simplerenv/get_results.py
```

## Sharing an existing SimplerEnv venv (alternative setup)

Setting up SimplerEnv from scratch is painful. If you already have a working SimplerEnv venv, you can install UniVLA's Python dependencies into it instead of maintaining a second environment — the torch stack and SimplerEnv's tensorflow/jax stack coexist fine.

```shell
SIMPLER_VENV=/path/to/SimplerEnv/.venv

# UniVLA deps
uv pip install --python ${SIMPLER_VENV}/bin/python \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python ${SIMPLER_VENV}/bin/python \
  transformers==4.44.0 tiktoken==0.6.0 pillow numpydantic==1.6.8
uv pip install --python ${SIMPLER_VENV}/bin/python \
  flash-attn==2.8.3 --no-build-isolation
```

If your SimplerEnv venv is pinned to an older torch (e.g. `torch 2.2.0+cu121`), use a flash-attn version that matches it (`flash-attn==2.6.3`) rather than forcing torch 2.4.0, which may fight with SimplerEnv's transitive pins.

When running evaluation, activate the SimplerEnv venv and put both repos on `PYTHONPATH`:

```shell
source ${SIMPLER_VENV}/bin/activate
export UNIVLA_ROOT=/path/to/UniVLA
export PYTHONPATH=${UNIVLA_ROOT}:${UNIVLA_ROOT}/reference/Emu3:${UNIVLA_ROOT}/reference/RoboVLMs:$PYTHONPATH
```

`scripts/bridge_univla.bash` already exports `PYTHONPATH` using `UNIVLA_ROOT` (falling back to its own location), so running the script from an activated SimplerEnv venv is enough — no manual export needed.

## Known issues and fixes

The upstream evaluation scripts assume the original author's filesystem layout and have a few paths hardcoded. If you run into these, here's what to patch:

1. **Missing `robovlms` package.** `reference/RoboVLMs/` only contains the UniVLA-specific pieces (`configs/`, `eval/`, `scripts/`). The `robovlms/` Python package itself must be cloned from upstream:
   ```shell
   git clone https://github.com/Robot-VLAs/RoboVLMs.git /tmp/robovlms
   mv /tmp/robovlms/robovlms reference/RoboVLMs/robovlms
   ```

2. **`from turtle import pd` in `robovlms/data/vid_llava_dataset.py`.** An upstream bug (IDE autocomplete accident) that pulls in `tkinter` and crashes the import chain on systems without it. `pd` is unused — just delete the line.

3. **`transformers.processing_utils` missing `ProcessingKwargs`.** Requires `transformers>=4.44.0`. Upgrade with `uv pip install --python .../python transformers==4.44.0`.

4. **flash-attn `undefined symbol: _ZNK3c105Error4whatEv`.** C++ ABI mismatch between the flash-attn prebuilt wheel and the installed torch. Rebuild flash-attn against the actually-installed torch (`--no-build-isolation`), or downgrade flash-attn to a version matching your torch.

5. **`PermissionError: '/share'` on startup.** `eval/simpler/main_inference_emu.py:151` defaults `--CACHE_ROOT` to a path on the original author's cluster. Pass `--CACHE_ROOT ./logs/simpler` (or any writable path) when invoking the script.

6. **`HFValidationError: '/share/project/yuqi.wang/OmniSim/pretrain/Emu3-Base'`.** Two more hardcoded defaults in `main_inference_emu.py`:
   - `--vq_hub` is misleadingly named: it actually loads the Emu3 *text* tokenizer (`Emu3Tokenizer`), not the VQ model. Your finetuned checkpoint already contains the necessary tokenizer files (`emu3.tiktoken`, `emu3_vision_tokens.txt`, `tokenizer_config.json`, `special_tokens_map.json`), so pass `--vq_hub ${CKPT_PATH}`.
   - `--vision_hub` loads the real VQ vision tokenizer, which is a separate HuggingFace model. Download it with:
     ```shell
     hf download BAAI/Emu3-VisionTokenizer \
       --local-dir /path/to/UniVLA/pretrain/Emu3-VisionTokenizer
     ```
     then pass `--vision_hub /path/to/UniVLA/pretrain/Emu3-VisionTokenizer`.

7. **Hardcoded FAST action tokenizer path in `eval/simpler/model_wrapper.py`.** Originally pointed to `/share/project/yuqi.wang/UniVLA/pretrain/fast_*`. The repo ships these tokenizers under `pretrain/` (`fast_bridge_t5_s50`, `fast_google_a5_s50`, `fast`). Either hardcode the absolute local path or read from an env var like `UNIVLA_FAST_ROOT`.

`scripts/bridge_univla.bash` in this repo already has these fixes wired in (it passes `--vq_hub $ckpt_dir`, `--vision_hub $vision_hub`, and `--CACHE_ROOT ./logs/simpler`), so once the packages and downloads above are in place, `bash scripts/bridge_univla.bash ${CKPT_PATH}` should run end-to-end.

