# 🌐 UniVLA: Unified Vision-Language-Action Model

A general-purpose **VLA Model** designed to unify **vision, language, and action** for robotics and autonomous driving.

<img src="docs/imgs/univla.png" alt="UniVLA" height="300">

> 📜 [[technical report](https://arxiv.org/abs/2506.19850)] 🤗 [[model weights](https://huggingface.co/Yuqi1997/UniVLA)] 🤖 [[project page](https://robertwyq.github.io/univla.github.io)]

## 🚀 News
- **2025.10.15**: DriveVLA-W0, extended the UniVLA framework to autonomous driving, [code](https://github.com/BraveGroup/DriveVLA-W0) released.
- **2025.6.27**: code released for robotic simulations.
- **2025.6.25**: paper released on the arXiv.

## 🧪 Highlights
- [x] **Unified Vision-Language-Action Model**: supports image grounding, video generation, and action prediction.
- [x] **Strong Performance on Several Robotics Benchmarks**: support CALVIN, LIBERO, SimplerEnv.
- [x] **Interleaved Video Training**: support interleaved vision-action training in Markov Decision Process.
- [x] **Broader Applications**: Real-robot ALOHA & Autonomous Driving.

## 🔧 REPO TODO List
- [x] Policy learning for CALVIN, LIBERO, and SimplerEnv.
- [x] Support for evaluation.
- [x] World model pretraining for video generation.
- [x] Example for real-robot ALOHA.
- [x] Support for autonomous driving.
- [ ] Support for general grounding.

## 📚 Experiments

### Emu3 Pretraining Models
You can download the pretraining models from HuggingFace, here we provide the links.

> [Emu3-base](https://huggingface.co/BAAI/Emu3-Stage1)

> [Emu3-vision](https://huggingface.co/BAAI/Emu3-VisionTokenizer)

### World Model Training
More details can be found in the [World Model Training](docs/world_model.md) document.
```shell
# train the world model
bash scripts/pretrain/train_video_1node.sh
```
> [world model pretraining ckpts](https://huggingface.co/Yuqi1997/UniVLA/tree/main/WORLD_MODEL_POSTTRAIN)

This model is used to serve as the prerained model for the downstream policy learning tasks, such as CALVIN, LIBERO, and SimplerEnv.

### 1. CALVIN Benchmark
| Method | Mode  | Setting                                      | AVG  | CKPT |
|--------|-------|----------------------------------------------|------|------|
| UniVLA   | video sft | ABCD->D       | 4.63 (5x:4.71) | [huggingface](https://huggingface.co/Yuqi1997/UniVLA/tree/main/UNIVLA_CALVIN_ABCD_VIDEO_BS192_8K)  |
> **Note:** 5× means 5× inference steps, i.e., 180 steps total.

#### Training
- Here provide single node training script, recommend multi-node training.
```shell
# video sft
bash scripts/simulator/calvin/train_calvin_abcd_video.sh
```
### 2. LIBERO Benchmark
| Method | Mode  | SPATIAL | OBJECTS | GOAL  | 10   | AVG   | CKPT |
|--------|-------|---------|---------|-------|------|-------| -----|
| UniVLA   | img sft  | 97.0    | 99.0    | 92.6  | 90.8 | 94.8  | [huggingface](https://huggingface.co/Yuqi1997/UniVLA/tree/main/UNIVLA_LIBERO_IMG_BS192_8K)  |
| UniVLA   | video sft  | 95.4    | 98.8    | 93.6  | 94.0 | 95.5  |  [huggingface](https://huggingface.co/Yuqi1997/UniVLA/tree/main/UNIVLA_LIBERO_VIDEO_BS192_8K) |

#### Training
```shell
bash scripts/simulator/libero/train_libero_video.sh
```

### 3. SimplerEnv Benchmark
| Method | Robot |Mode  | Put Spoon | Put Carrot | Stack Block  | Put Eggplant   | AVG   | CKPT |
|--------|-------|-------| -----------|------------|--------------|----------------|-------| -----|
| UniVLA   | Bridge(WidowX) | video sft  | 83.3    | 66.7   | 33.3  | 95.8 | 69.8  |   [huggingface](https://huggingface.co/Yuqi1997/UniVLA/tree/main/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K) |

#### Training
```shell
bash scripts/simulator/simplerenv/train_simplerenv_bridge_video.sh
```

## Setup
> Here we use [uv](https://docs.astral.sh/uv/) to manage the Python environment.
```shell
uv sync
uv pip install flash-attn==2.8.3 --no-build-isolation
```
### Benchmark setup, training and evaluation
- [CALVIN](docs/calvin.md)
- [LIBERO](docs/libero.md)
- [SimplerEnv](docs/simplerenv.md)
- [ALOHA](docs/aloha.md)

<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-4">📁 Code Structure</h2>
    <pre style="background-color: #f9f9f9; padding: 1.25rem; border-radius: 8px; font-size: 14px; overflow-x: auto;">
<span style="color: #6c757d;">OmniSim/</span>
├── <strong>configs/</strong>       <span style="color: #6c757d;"># Model configuration files</span>
├── <strong>models/</strong>        <span style="color: #6c757d;"># Tokenizer and diffusion test</span>
├── <strong>train/</strong>         <span style="color: #6c757d;"># Training dataset and pipeline</span>
├── <strong>reference/</strong>     <span style="color: #6c757d;"># Reference code</span>
│   ├── <strong>Emu3/</strong>      <span style="color: #6c757d;"># Base code</span>
│   └── <strong>RoboVLMs/</strong>  <span style="color: #6c757d;"># Evaluation code</span>
├── <strong>scripts/</strong>       <span style="color: #6c757d;"># Shell scripts for training & evaluation</span>
├── <strong>tools/</strong>         <span style="color: #6c757d;"># Data preprocessing tools</span>
└── <strong>README.md</strong>      <span style="color: #6c757d;"># Project description and user guide</span>
    </pre>
  </div>
</section>

## ❤️ Acknowledgement
Our work is built upon the following projects, Thanks for their great open-source work!
- [Emu3](https://github.com/baaivision/Emu3)
- [RoboVLMs](https://github.com/Robot-VLAs/RoboVLMs)
- [OpenVLA](https://github.com/openvla/openvla)

## 🌟Citation
If you find this project useful, please consider citing our work:
```bibtex
@article{wang2025unified,
  title={Unified Vision-Language-Action Model},
  author={Wang, Yuqi and Li, Xinghang and Wang, Wenxuan and Zhang, Junbo and Li, Yingyan and Chen, Yuntao and Wang, Xinlong and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2506.19850},
  year={2025}
}
```
