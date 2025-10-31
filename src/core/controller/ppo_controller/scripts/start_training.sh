#!/bin/bash

# PPO训练快速启动脚本

echo "Starting PPO Training in ROS Motion Planning Environment"
echo "======================================================="

# 检查环境
source /home/galbot/ros_motion_planning/devel/setup.bash

# 检查Python依赖
echo "Checking Python dependencies..."
python3 -c "import stable_baselines3; print('✓ stable-baselines3 available')" 2>/dev/null || {
    echo "❌ stable-baselines3 not found. Installing..."
    pip3 install stable-baselines3[extra]
}

python3 -c "import tensorboard; print('✓ tensorboard available')" 2>/dev/null || {
    echo "❌ tensorboard not found. Installing..."
    pip3 install tensorboard
}

python3 -c "import gymnasium; print('✓ gymnasium available')" 2>/dev/null || {
    echo "❌ gymnasium not found. Installing..."
    pip3 install gymnasium
}

# 创建必要目录
echo "Creating directories..."
rosrun ppo_controller create_dirs.sh

# 启动训练
echo ""
echo "Starting PPO training..."
echo "- Simulation: warehouse world"
echo "- Episodes: 100"
echo "- TensorBoard: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop training"
echo ""

# 启动训练（前台运行）
roslaunch ppo_controller train_ppo_simulation.launch \
    world:=warehouse \
    map:=warehouse \
    training_episodes:=100 \
    timesteps:=50000 \
    model_save_path:="$(rospack find ppo_controller)/models/warehouse_training_model.zip"