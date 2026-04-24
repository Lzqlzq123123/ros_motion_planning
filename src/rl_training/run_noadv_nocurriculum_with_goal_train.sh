#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/lzq/ros_motion_planning"
CONFIG="$ROOT/src/rl_training/config/forklift_td3_noadv_nocurriculum_with_goal.yaml"

source /opt/ros/noetic/setup.bash
source /data/lzq/miniconda3/etc/profile.d/conda.sh
conda activate rl
source "$ROOT/devel/setup.bash"

python "$ROOT/src/rl_training/train_velodyne_td3_with_goal.py" \
  --config "$CONFIG" \
  --experiment_name forklift_movebase_with_goal \
  --run_name run_noadv_nocurr_retrain_seed3 \
  --seed 3 \
  --max_timesteps 5000000 \
  --eval_freq 5000 \
  --eval_episodes 10
