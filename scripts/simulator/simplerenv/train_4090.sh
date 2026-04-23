export CUDA_VISIBLE_DEVICES=0
NGPUS=1

DATAPATH="$HOME/data/simplerenv_bridge_trainval.h5"
ACTION_TOKENIZER_PATH="$HOME/projects/UniVLA/pretrain/fast_bridge_t5_s50"
EXP_NAME="UNIVLA_4090_SMOKE"

export PYTHONPATH=$(pwd)
export DS_SKIP_CUDA_CHECK=1
export LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export WANDB_MODE=disabled


$HOME/projects/UniVLA/.venv/bin/python \
    train/train_moe.py \
    --model_config_path $HOME/projects/UniVLA/configs/moe_fast_video_2b.json \
    --from_scratch True \
    --output_dir "logs/"${EXP_NAME} \
    --optim sgd \
    --learning_rate 8e-5 \
    --null_prompt_prob 0.15 \
    --weight_decay 0.0 \
    --min_learning_rate 5e-6 \
    --max_grad_norm 2.0 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 20 \
    --dataloader_num_workers 2 \
    --dataloader_prefetch_factor 2 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 0 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --torch_compile False \
    --dataloader_persistent_workers False \
    --frames 2 \
    --action_frames 5 \
    --max_position_embeddings 2400 \
    --seed 42 \
    --logging_steps 5 \
    --gradient_checkpointing True \
    --save_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action True \
    --actions True \
    --actions_format "fast" \
    --use_gripper False \
    --video_format "interleave" \
    --report_to none \
    --run_name ${EXP_NAME} \
    --action_tokenizer_path ${ACTION_TOKENIZER_PATH} \
    --use_liger_kernel True \
