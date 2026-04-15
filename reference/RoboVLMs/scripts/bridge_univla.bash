# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth
UNIVLA_ROOT=${UNIVLA_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"}
export PYTHONPATH=${UNIVLA_ROOT}:${UNIVLA_ROOT}/reference/Emu3:${UNIVLA_ROOT}/reference/RoboVLMs:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

policy_model=openvla

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

ckpt_dir=$1
vision_hub=${VISION_HUB:-/home/zijian/projects/UniVLA/pretrain/Emu3-VisionTokenizer}
model_name=${MODEL_NAME:-emu_vla}

python eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --vq_hub $ckpt_dir --vision_hub $vision_hub \
  --CACHE_ROOT ./logs/simpler \
  --model_name ${model_name} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

python eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --vq_hub $ckpt_dir --vision_hub $vision_hub \
  --CACHE_ROOT ./logs/simpler \
  --model_name ${model_name} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

python eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --vq_hub $ckpt_dir --vision_hub $vision_hub \
  --CACHE_ROOT ./logs/simpler \
  --model_name ${model_name} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

python eval/simpler/main_inference_emu.py --policy-model ${policy_model} --emu_hub $ckpt_dir \
  --vq_hub $ckpt_dir --vision_hub $vision_hub \
  --CACHE_ROOT ./logs/simpler \
  --model_name ${model_name} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

UNIVLA_CKPT_NAME=${model_name} python eval/simpler/get_results.py