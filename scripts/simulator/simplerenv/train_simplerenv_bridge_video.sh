WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}
export CUDA_VISIBLE_DEVICES=4,5,6,7
NGPUS=4

DATAPATH="$HOME/data/simplerenv_bridge_trainval.h5"
ACTION_TOKENIZER_PATH="$HOME/projects/UniVLA/pretrain/fast_bridge_t5_s50"
EXP_NAME="UNIVLA_SIMPLERENV_BRIDGE_VIDEO_BS128_20k_STAGE1"

export WANDB_PROJECT="UniVLA"
export PYTHONPATH=$(pwd)
export DS_SKIP_CUDA_CHECK=1
export LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

# export WANDB_MODE=offline

$HOME/projects/UniVLA/.venv/bin/torchrun \
    --nproc_per_node=${NGPUS} \
    --nnodes=1 \
    --node_rank=${RANK} \
    train/train_moe.py \
    --model_name_or_path $HOME/data/Emu3-Stage1 \
    --model_config_path $HOME/projects/UniVLA/configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero2.json \
    --output_dir "logs/"${EXP_NAME} \
    --learning_rate 8e-5 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.1 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 2.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 20000 \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 500 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --torch_compile_mode "reduce-overhead" \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 4 \
    --frames 2 \
    --action_frames 5 \
    --max_position_embeddings 2400 \
    --seed 42 \
    --logging_steps 20 \
    --gradient_checkpointing True \
    --save_strategy steps \
    --save_steps 4000 \
    --save_only_model True \
    --eval_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action True \
    --actions True \
    --actions_format "fast" \
    --use_gripper False \
    --video_format "interleave" \
    --report_to wandb \
    --run_name ${EXP_NAME} \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
