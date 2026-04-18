#!/bin/bash
#SBATCH --account=fllm
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --time=0-23:59:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

bash scripts/simulator/simplerenv/train_simplerenv_bridge_video_a100.sh
