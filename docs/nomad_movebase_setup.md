# NoMaD + move_base 从零到跑通安装清单

这份清单的目标是让 `visualnav-transformer` 的 NoMaD 对抗轨迹生成服务，与 `ros_motion_planning` 的 `move_base` 仿真环境同时运行。

核心原则只有一条：

- 不要把两边依赖塞进同一个 Python 环境。
- `ros_motion_planning` 使用系统 `ROS Noetic + catkin`。
- `visualnav-transformer` 使用独立 `conda` 环境 `nomad_train`。
- 两边通过 ROS service `/nomad/make_plan` 连接。

补充说明：

- 上面这条仍然是默认推荐方案，适合迁移到新机器时照着做。
- 但在当前这台机器上，已经验证过 `conda rl` 也可以启动 `nomad_plan_service.py`。
- 前提是先 `source /opt/ros/noetic/setup.bash` 和 `source /data/lzq/ros_motion_planning/devel/setup.bash`，再激活 `rl`，并补齐 `PYTHONPATH`。
- 所以这份文档保留“独立环境”为标准方案，同时在后面给出“本机合并环境”的可选跑法。

---

## 1. 适用范围

- Ubuntu `20.04`
- ROS `Noetic`
- NVIDIA GPU 可用
- 已安装 Miniconda 或 Anaconda

如果不是 Ubuntu 20.04，先不要继续折腾原生环境，优先改走 Docker。

---

## 2. 目录约定

下面所有命令默认使用这组路径：

```bash
export RMP_ROOT=/data/lzq/ros_motion_planning
export VINT_ROOT=/data/lzq/visualnav-transformer
export CONDA_HOME=/data/lzq/miniconda3
export NOMAD_ENV=nomad_train
```

如果你的仓库不在这两个目录，先把变量改掉。

---

## 3. 先做基础检查

```bash
lsb_release -a
python3 --version
conda --version
nvidia-smi
```

至少要满足：

- `Ubuntu 20.04`
- `conda` 可用
- `nvidia-smi` 能正常输出

---

## 4. 从零安装 ROS Noetic

如果 `/opt/ros/noetic` 已经存在，可以跳过这一节。

### 4.1 添加 ROS 软件源

```bash
sudo apt update
sudo apt install -y curl gnupg2 lsb-release ca-certificates software-properties-common
sudo add-apt-repository universe -y

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
  | sudo tee /etc/apt/sources.list.d/ros1.list > /dev/null
```

### 4.2 安装 ROS Noetic

```bash
sudo apt update
sudo apt install -y \
  ros-noetic-desktop-full \
  python3-rosdep \
  python3-rosinstall \
  python3-rosinstall-generator \
  python3-wstool \
  build-essential
```

### 4.3 初始化 rosdep

```bash
sudo rosdep init || true
rosdep update
```

### 4.4 写入 shell

```bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source /opt/ros/noetic/setup.bash
```

---

## 5. 克隆两个仓库

如果仓库已经存在，可以跳过。

```bash
mkdir -p /data/lzq
cd /data/lzq

git clone <your_ros_motion_planning_repo_url> ros_motion_planning
git clone <your_visualnav_transformer_repo_url> visualnav-transformer
```

---

## 6. 安装 `ros_motion_planning` 侧依赖

### 6.1 系统依赖

```bash
sudo apt update
sudo apt install -y \
  git \
  python3-pip \
  python-is-python3 \
  python3-catkin-tools \
  ros-noetic-amcl \
  ros-noetic-base-local-planner \
  ros-noetic-map-server \
  ros-noetic-move-base \
  ros-noetic-navfn \
  ros-noetic-cv-bridge \
  libgoogle-glog-dev
```

### 6.2 安装 Conan 1.x

不要装 Conan 2.x，这个仓库构建脚本按 `1.59.0` 写的。

```bash
pip3 install conan==1.59.0
conan remote add conancenter https://center.conan.io || true
```

### 6.3 编译 `ros_motion_planning`

```bash
export RMP_ROOT=/data/lzq/ros_motion_planning

source /opt/ros/noetic/setup.bash
cd $RMP_ROOT/scripts
./build.sh
```

编译完成后确认下面两个文件存在：

```bash
ls $RMP_ROOT/devel/setup.bash
ls $RMP_ROOT/devel/lib/python3/dist-packages
```

---

## 7. 安装 `visualnav-transformer` 侧环境

这里使用仓库中已经存在的训练环境文件 `train/train_environment.yml`，环境名默认就是 `nomad_train`。

### 7.1 创建 Conda 环境

```bash
export VINT_ROOT=/data/lzq/visualnav-transformer

cd $VINT_ROOT
conda env create -f train/train_environment.yml
```

### 7.2 激活环境

```bash
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate nomad_train
```

### 7.3 安装本地包

```bash
cd $VINT_ROOT
pip install -e train/
pip install -e diffusion_policy/
```

### 7.4 验证 Python 依赖

```bash
python -c "import PIL, torch, torchvision, cv2, diffusers; print('python=', __import__('sys').executable)"
```

---

## 8. 准备 NoMaD 模型与参数文件

### 8.1 至少确认下面两个文件存在

```bash
ls $VINT_ROOT/train/config/nomad.yaml
ls $VINT_ROOT/train/logs/nomad/ema_latest.pth
```

如果第二个文件不存在，先把你自己的 NoMaD 权重放到这里，或者改下面参数文件里的 `weights_path`。

### 8.2 推荐的对抗轨迹参数文件

如果你要跑“目标条件 + 多样本 + 对抗筛选”，直接用：

```bash
$VINT_ROOT/deployment/config/nomad_service_params_goal_multisample.yaml
```

这个文件已经默认开启：

- `adversarial_selection_enabled: true`
- `num_adversarial_samples: 96`
- `service_name: /nomad/make_plan`

如果你要跑探索式对抗轨迹，改用：

```bash
$VINT_ROOT/deployment/config/nomad_service_params_explore_adversarial.yaml
```

---

## 9. 确认 `move_base` 侧已经切到 `diffusion`

执行前先看一眼：

```bash
grep -n "robot2_global_planner" $RMP_ROOT/src/user_config/user_config.yaml
```

你希望看到的是：

```yaml
robot2_global_planner: diffusion
```

如果不是，就改成 `diffusion`。

---

## 10. 启动顺序

必须按这个顺序启动：

1. 先启动 `ros_motion_planning` 仿真环境
2. 再启动 `NoMaD` 服务
3. 等 `/nomad/make_plan` 就绪后，再在 RViz 发目标

不要反过来。

---

## 11. 终端 A：启动 `move_base` 仿真环境

```bash
export RMP_ROOT=/data/lzq/ros_motion_planning

source /opt/ros/noetic/setup.bash
cd $RMP_ROOT
source devel/setup.bash

cd $RMP_ROOT/scripts
./main.sh
```

如果 `main.sh` 启动成功，你应该能看到 Gazebo、RViz 和 `move_base` 相关节点起来。

---

## 12. 终端 B：启动 NoMaD 对抗轨迹服务

下面这段命令直接按原样跑，顺序不要改：

```bash
export RMP_ROOT=/data/lzq/ros_motion_planning
export VINT_ROOT=/data/lzq/visualnav-transformer
export CONDA_HOME=/data/lzq/miniconda3

source /opt/ros/noetic/setup.bash
cd $RMP_ROOT
source devel/setup.bash

source $CONDA_HOME/etc/profile.d/conda.sh
conda activate nomad_train

export PYTHONNOUSERSITE=1
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$RMP_ROOT/devel/lib/python3/dist-packages:$PYTHONPATH
export VISUALNAV_ROOT=$VINT_ROOT

cd $VINT_ROOT
python deployment/src/nomad_plan_service.py \
  _params_file:=deployment/config/nomad_service_params_goal_multisample.yaml
```

看到下面这类日志再继续：

```text
Ready on /nomad/make_plan
```

### 12.1 本机可选：复用 `conda rl`

如果你就是在当前这台机器 `/data/lzq` 上运行，而且已经确认 `rl_environment.yaml` 里的依赖和 `visualnav-transformer` 兼容，也可以不用 `nomad_train`，直接复用 `rl` 环境：

```bash
export RMP_ROOT=/data/lzq/ros_motion_planning
export VINT_ROOT=/data/lzq/visualnav-transformer

source /opt/ros/noetic/setup.bash
cd $RMP_ROOT
source devel/setup.bash

source /data/lzq/miniconda3/etc/profile.d/conda.sh
conda activate rl

export PYTHONNOUSERSITE=1
export VISUALNAV_ROOT=$VINT_ROOT
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$RMP_ROOT/devel/lib/python3/dist-packages:$VINT_ROOT:$VINT_ROOT/train:$VINT_ROOT/deployment/src:$VINT_ROOT/diffusion_policy:$PYTHONPATH

cd $VINT_ROOT
python deployment/src/nomad_plan_service.py \
  _params_file:=deployment/config/nomad_service_params_goal_multisample.yaml
```

这个跑法已经在当前机器上验证过模块导入可用，但不建议把它当成跨机器默认方案。

---

## 13. 终端 C：做连通性检查

### 13.1 检查服务

```bash
source /opt/ros/noetic/setup.bash
cd $RMP_ROOT
source devel/setup.bash

rosservice list | grep nomad
```

应该至少看到：

```bash
/nomad/make_plan
```

### 13.2 检查机器人话题

```bash
rostopic list | grep /robot2/
```

至少要能看到这类话题：

- `/robot2/camera/rgb/image_raw`
- `/robot2/camera/rgb/camera_info`
- `/robot2/ground_truth/state`

### 13.3 检查规划路径输出

```bash
rostopic echo -n 1 /robot2/move_base/PathPlanner/plan
```

---

## 14. 开始跑

1. 打开 RViz
2. 使用 `2D Nav Goal`
3. 给 `robot2` 发目标
4. 观察 NoMaD 服务是否返回路径
5. 观察 `move_base` 是否开始跟踪全局路径

---

## 15. 停止命令

### 15.1 关闭 NoMaD 服务

在 NoMaD 所在终端直接 `Ctrl+C`。

### 15.2 关闭仿真

```bash
cd $RMP_ROOT/scripts
./killpro.sh
```

---

## 16. 常见故障排查

### 16.1 `No module named rospy`

原因：

- 你先激活了 `conda`，但没有先 `source /opt/ros/noetic/setup.bash`

修复：

```bash
source /opt/ros/noetic/setup.bash
cd $RMP_ROOT
source devel/setup.bash
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate nomad_train
```

### 16.2 `No module named nomad_planner_msgs`

原因：

- `ros_motion_planning` 没编过，或者没有 `source devel/setup.bash`

修复：

```bash
cd $RMP_ROOT/scripts
./build.sh

cd $RMP_ROOT
source devel/setup.bash
```

### 16.3 `NoMaD weights file not found`

原因：

- 权重文件不在参数文件指定的位置

修复：

- 把权重放到 `$VINT_ROOT/train/logs/nomad/ema_latest.pth`
- 或者修改 `_params_file` 中的 `weights_path`

### 16.4 `Timed out waiting for NoMaD service`

原因：

- `move_base` 已启动，但 `nomad_plan_service.py` 没起来
- 或者 `service_name` 对不上

修复：

- 确认终端 B 日志里有 `Ready on /nomad/make_plan`
- 确认参数文件里的 `service_name` 也是 `/nomad/make_plan`

### 16.5 `move_base` 没有调用 NoMaD

原因：

- `robot2_global_planner` 不是 `diffusion`

修复：

```bash
grep -n "robot2_global_planner" $RMP_ROOT/src/user_config/user_config.yaml
```

如果不是 `diffusion`，改掉后重启 `main.sh`。

### 16.6 能出路径，但机器人不动

优先检查：

- `/robot2/move_base/PathPlanner/plan` 是否有路径
- `/cmd_vel` 是否有速度输出
- 局部规划器和控制器是否正常启动

---

## 17. 最短路径版本

如果你的机器已经装好 ROS Noetic 和 Miniconda，最短只需要做这几步：

```bash
# 1. 编译 ros_motion_planning
source /opt/ros/noetic/setup.bash
cd /data/lzq/ros_motion_planning/scripts
./build.sh

# 2. 建 NoMaD 环境
cd /data/lzq/visualnav-transformer
conda env create -f train/train_environment.yml
source /data/lzq/miniconda3/etc/profile.d/conda.sh
conda activate nomad_train
pip install -e train/
pip install -e diffusion_policy/

# 3. 终端 A 启动仿真
source /opt/ros/noetic/setup.bash
cd /data/lzq/ros_motion_planning
source devel/setup.bash
./scripts/main.sh

# 4. 终端 B 启动 NoMaD 服务
source /opt/ros/noetic/setup.bash
cd /data/lzq/ros_motion_planning
source devel/setup.bash
source /data/lzq/miniconda3/etc/profile.d/conda.sh
conda activate nomad_train
export PYTHONNOUSERSITE=1
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:/data/lzq/ros_motion_planning/devel/lib/python3/dist-packages:$PYTHONPATH
export VISUALNAV_ROOT=/data/lzq/visualnav-transformer
cd /data/lzq/visualnav-transformer
python deployment/src/nomad_plan_service.py \
  _params_file:=deployment/config/nomad_service_params_goal_multisample.yaml
```

---

## 18. 建议

- 第一次跑通之前，不要同时改 `user_config.yaml`、NoMaD 参数文件和模型权重。
- 先用默认 `goal_multisample` 参数文件跑通，再调 `adversarial_*` 超参数。
- 如果后面要批量评估，可以再把这套流程封装成 `tmux` 脚本。

---

## 19. 本次实际运行命令

下面这条命令就是本次实际用于复查 `扩散 + movebase` 对抗验证的统一评测命令。

它会自动完成这些动作：

- 改写 `robot2_global_planner=diffusion`
- 重启 `scripts/main.sh`
- 启动 `nomad_plan_service.py`
- 使用 `rl` 环境执行评测脚本
- 输出结果到单独目录

```bash
source /opt/ros/noetic/setup.bash
source /data/lzq/miniconda3/etc/profile.d/conda.sh
conda activate rl
source /data/lzq/ros_motion_planning/devel/setup.bash

python /data/lzq/ros_motion_planning/src/rl_training/model_screening/run_table31_movebase_eval.py \
  --output-dir /data/lzq/ros_motion_planning/src/rl_training/model_screening/recheck_ours_diffusion_20260508 \
  --methods ours_diffusion \
  --episodes 10 \
  --goal-offset 0.0 \
  --nomad-classical-guidance-weight 0.15 \
  --nomad-opponent-guidance-enabled true \
  --nomad-num-adversarial-samples 128 \
  --nomad-adversarial-max-step 0.25 \
  --nomad-adversarial-max-turn-deg 60.0 \
  --nomad-adversarial-guidance-max-deviation 2.0 \
  --nomad-adversarial-guidance-mean-deviation 1.2 \
  --execute
```

对应输出目录：

```bash
/data/lzq/ros_motion_planning/src/rl_training/model_screening/recheck_ours_diffusion_20260508
```

主要结果文件：

- `table3_eval_pretty.json`
- `table3_eval_summary.csv`
- `csv/ours_diffusion.csv`
- `stack_logs/ours_diffusion.log`
- `stack_logs/ours_diffusion__nomad_service.log`

---

## 20. 手动启动方式

如果你不想走统一评测脚本，也可以手动拆成 3 个终端。

### 20.1 终端 A：启动仿真栈

```bash
export RMP_ROOT=/data/lzq/ros_motion_planning

source /opt/ros/noetic/setup.bash
cd $RMP_ROOT
source devel/setup.bash

cd $RMP_ROOT/scripts
./main.sh
```

### 20.2 终端 B：启动 NoMaD 服务

如果你要复现本次使用的对抗筛选配置，建议先生成一个临时参数文件，内容如下：

```yaml
service_name: /nomad/make_plan
camera_topic: /robot2/camera/rgb/image_raw
camera_info_topic: /robot2/camera/rgb/camera_info
robot_pose_topic: /robot2/ground_truth/state
plan_topic: /robot2/move_base/PathPlanner/plan
plan_overlay_topic: plan_overlay
camera_height: 0.95
camera_x_offset: 0.45
waypoint_spacing: 2
metric_waypoint_spacing: 0.11
goal_image_dir: deployment/goal
default_goal_image: goal.jpg
models_catalog_path: deployment/config/models.yaml
model_name: diffusion
config_path: train/config/nomad.yaml
weights_path: train/logs/nomad/ema_latest.pth
opponent_guidance_enabled: true
opponent_pose_topic: /robot1/ground_truth/state
opponent_goal_topic: /robot1/rl_goal
opponent_min_separation: 0.6
opponent_guidance_gain: 0.3
opponent_max_offset: 0.5
opponent_prediction_horizon: 5.0
classical_guidance_enabled: false
classical_map_topic: /map
classical_global_plan_service_enabled: false
classical_global_plan_service: /robot2/move_base/PathPlanner/make_plan
classical_global_plan_service_timeout: 0.2
classical_global_plan_tolerance: 0.0
classical_global_plan_enabled: false
classical_global_plan_topic: /robot2/move_base/PathPlanner/plan
classical_obstacle_threshold: 50
classical_unknown_is_obstacle: true
classical_guidance_mode: blend
classical_guidance_weight: 0.15
classical_max_expansions: 20000
adversarial_selection_enabled: true
num_adversarial_samples: 128
adversarial_collision_radius: 0.55
adversarial_time_decay: 0.9
adversarial_max_step: 0.25
adversarial_max_turn_deg: 60.0
adversarial_guidance_filter_enabled: false
adversarial_guidance_max_deviation: 2.0
adversarial_guidance_mean_deviation: 1.2
ego_speed_estimate: 0.3
```

然后启动：

```bash
export RMP_ROOT=/data/lzq/ros_motion_planning
export VINT_ROOT=/data/lzq/visualnav-transformer
export CONDA_HOME=/data/lzq/miniconda3

source /opt/ros/noetic/setup.bash
cd $RMP_ROOT
source devel/setup.bash

source $CONDA_HOME/etc/profile.d/conda.sh
conda activate nomad_train

export PYTHONNOUSERSITE=1
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$RMP_ROOT/devel/lib/python3/dist-packages:$PYTHONPATH
export VISUALNAV_ROOT=$VINT_ROOT

cd $VINT_ROOT
python deployment/src/nomad_plan_service.py \
  _params_file:=/tmp/nomad_params_recheck.yaml
```

如果你不需要复现这次的参数，只想用仓库默认参数，那么把 `_params_file` 改成：

```bash
_params_file:=deployment/config/nomad_service_params_goal_multisample.yaml
```

### 20.3 终端 C：启动评测

```bash
source /opt/ros/noetic/setup.bash
source /data/lzq/miniconda3/etc/profile.d/conda.sh
conda activate rl
source /data/lzq/ros_motion_planning/devel/setup.bash

python /data/lzq/ros_motion_planning/src/rl_training/eval_velodyne_td3_with_goal.py \
  --config /data/lzq/ros_motion_planning/src/rl_training/model_screening/recheck_ours_diffusion_20260508/configs/ours_diffusion.yaml \
  --robot1_mode movebase \
  --opponent_mode diffusion \
  --episodes 10 \
  --csv_path /data/lzq/ros_motion_planning/src/rl_training/model_screening/recheck_ours_diffusion_20260508/csv/ours_diffusion.csv
```

---

## 21. 如何确认启动成功

### 21.1 检查 NoMaD 服务

```bash
source /opt/ros/noetic/setup.bash
rosservice list | grep nomad
```

至少要出现：

```bash
/nomad/make_plan
```

### 21.2 检查对抗车是否在走 NoMaD 路径

```bash
rostopic echo -n 1 /robot2/move_base/PathPlanner/plan
```

有路径输出，说明 `robot2` 的 `move_base` 已经在消费 NoMaD 返回的全局轨迹。

### 21.3 检查评测结果

```bash
tail -n 20 /data/lzq/ros_motion_planning/src/rl_training/model_screening/recheck_ours_diffusion_20260508/csv/ours_diffusion.csv
```

最后一行 `row_type=summary` 就是本次统计结果。
