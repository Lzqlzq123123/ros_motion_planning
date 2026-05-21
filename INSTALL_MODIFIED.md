# 修改版 ros_motion_planning 安装说明

这份说明针对当前仓库 `/data/lzq/ros_motion_planning` 的实际结构，不是上游原始仓库说明。

当前仓库包含两套相互独立但会联动的环境：

1. `ROS Noetic + catkin + Conan`：用于仿真、`move_base`、插件和消息编译。
2. `conda rl`：用于强化学习训练与评估。

如果要跑 NoMaD 扩散规划，默认推荐第三套独立环境：

3. `visualnav-transformer + conda nomad_train`：用于 `nomad_plan_service.py`。

补充说明：

- 在这台机器上，已经验证 `conda rl` 也能跑 `nomad_plan_service.py`。
- 但前提是必须先 `source /opt/ros/noetic/setup.bash`，再 `source /data/lzq/ros_motion_planning/devel/setup.bash`，最后再激活 `rl`。
- 如果换到别的机器，默认还是先按独立 `nomad_train` 环境理解，不要直接假设 `rl` 一定可复用。

## 1. 适用范围

- Ubuntu 20.04
- ROS Noetic
- Python 3
- Conda 可用
- 如需 NoMaD 或训练加速，建议有 NVIDIA GPU

## 2. 仓库内现有说明

你可以结合这几份文档使用：

- `README.md`：基础编译与仿真入口
- `docs/nomad_movebase_setup.md`：NoMaD + move_base 联动
- `README_RSL_RL.md`：本地 `rsl_rl` 集成说明

这份文档的作用是把它们按当前仓库真实结构收口成一套可执行步骤。

## 3. 安装 ROS 侧依赖

先确保 ROS Noetic 已安装并可用：

```bash
source /opt/ros/noetic/setup.bash
```

安装系统依赖：

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

安装 Conan 1.x：

```bash
pip3 install conan==1.59.0
conan remote add conancenter https://center.conan.io || true
```

注意：不要替换成 Conan 2.x，这个仓库的构建脚本按 1.x 写的。

## 4. 编译当前仓库

```bash
cd /data/lzq/ros_motion_planning
source /opt/ros/noetic/setup.bash
cd scripts
./build.sh
```

`scripts/build.sh` 会执行两件事：

1. 在 `3rd/` 下执行 `conan install`
2. 回到仓库根目录执行 `catkin_make`

编译完成后确认：

```bash
ls /data/lzq/ros_motion_planning/devel/setup.bash
```

## 5. 启动基础仿真

```bash
cd /data/lzq/ros_motion_planning
source /opt/ros/noetic/setup.bash
source devel/setup.bash
cd scripts
./main.sh
```

关闭：

```bash
cd /data/lzq/ros_motion_planning/scripts
./killpro.sh
```

说明：

- `main.sh` 会根据 `src/user_config/user_config.yaml` 动态生成配置。
- 不要优先去改 launch 产物，先改 `src/user_config/user_config.yaml`。

## 6. 安装 RL 训练环境

当前仓库自带 Conda 环境文件：

- `rl_environment.yaml`

直接创建：

```bash
cd /data/lzq/ros_motion_planning
conda env create -f rl_environment.yaml
conda activate rl
```

这个环境主要用于：

- TD3 训练与评估
- `src/rl_training/train_new.py`
- `src/rl_training/eval_trained_policy.py`
- `src/rl_training/train_velodyne_td3.py`
- `src/rl_training/train_velodyne_td3_with_goal.py`
- `src/rl_training/eval_velodyne_td3_with_goal.py`

说明：

- 如果你只使用 TD3 训练或评估，不需要额外安装或单独配置 `rsl_rl`。
- 如果你使用 PPO 路线，才需要关心仓库内自带的本地 `rsl_rl` 修改版。

## 7. 本地 rsl_rl 的真实位置(可选)

这一节只和 PPO 路线有关。只跑 TD3 可以跳过。

当前仓库不是把自定义 `rsl_rl` 放在根目录 `third_party/`，而是放在：

```bash
src/rl_training/third_party/rsl_rl
```

关键文件：

```bash
src/rl_training/third_party/rsl_rl/rsl_rl/modules/res_actor_critic.py
```

PPO 训练和评估脚本现在会优先从这个目录加载本地修改版 `rsl_rl`，而不是依赖 Conda 环境里安装的版本。

## 8. RL 训练与评估

### 8.1 TD3 路线

如果你只跑 TD3，这条路线不依赖仓库内的自定义 `rsl_rl`。

训练入口：

```bash
cd /data/lzq/ros_motion_planning
conda activate rl
python src/rl_training/train_velodyne_td3.py \
  --config src/rl_training/config/forklift_td3_noadv_nocurriculum.yaml
```

带目标版本：

```bash
cd /data/lzq/ros_motion_planning
conda activate rl
python src/rl_training/train_velodyne_td3_with_goal.py \
  --config src/rl_training/config/forklift_td3_noadv_nocurriculum_with_goal.yaml
```

评估入口：

```bash
cd /data/lzq/ros_motion_planning
conda activate rl
python src/rl_training/eval_velodyne_td3_with_goal.py \
  --config src/rl_training/config/forklift_td3_noadv_nocurriculum_with_goal.yaml \
  --model_path src/rl_training/logs/forklift_movebase_with_goal/run_1/td3_model
```

### 8.2 PPO + rsl_rl 路线

这一条路线才会使用仓库内的自定义 `rsl_rl`。

训练：

```bash
cd /data/lzq/ros_motion_planning
conda activate rl
python src/rl_training/train_new.py \
  --config src/rl_training/config/forklift_ppo.yaml
```

评估：

```bash
cd /data/lzq/ros_motion_planning
conda activate rl
python src/rl_training/eval_trained_policy.py \
  --config src/rl_training/config/forklift_ppo.yaml \
  --model src/rl_training/logs/forklift_ppo/run_1/model_100.pt \
  --num-episodes 5
```

PPO 配置里已经指向自定义网络：

```yaml
runner:
  policy:
    class_name: "ResActorCritic"
```

## 9. NoMaD 扩散规划环境

如果你只是要跑 RL 训练，不需要这一节。

如果你要跑 `move_base + diffusion`，默认推荐使用独立的 `visualnav-transformer` 环境。

但对这台机器，已经实测存在第二种可行方式：直接复用 `conda rl`。

主说明文档：

- `docs/nomad_movebase_setup.md`

核心原则：

1. `ros_motion_planning` 继续使用 ROS/catkin 环境
2. 默认推荐 `visualnav-transformer` 使用独立 `nomad_train` 环境
3. 两边通过 `/nomad/make_plan` 服务连接

### 9.1 推荐跑法：独立 `nomad_train`

典型启动顺序：

1. 终端 A：启动 `ros_motion_planning/scripts/main.sh`
2. 终端 B：激活 `nomad_train`，启动 `visualnav-transformer/deployment/src/nomad_plan_service.py`
3. 等日志出现 `Ready on /nomad/make_plan`
4. 在 RViz 发送目标

### 9.2 本机已验证跑法：复用 `conda rl`

这台机器已经验证过，只要把 ROS 和 catkin 工作区先 source 进来，`rl` 环境也能导入 NoMaD 服务需要的模块。

终端 B 可以直接这样启动：

```bash
source /opt/ros/noetic/setup.bash
source /data/lzq/ros_motion_planning/devel/setup.bash
source /data/lzq/miniconda3/etc/profile.d/conda.sh
conda activate rl

export VISUALNAV_ROOT=/data/lzq/visualnav-transformer
export PYTHONNOUSERSITE=1
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:/data/lzq/ros_motion_planning/devel/lib/python3/dist-packages:/data/lzq/visualnav-transformer:/data/lzq/visualnav-transformer/train:/data/lzq/visualnav-transformer/deployment/src:/data/lzq/visualnav-transformer/diffusion_policy:$PYTHONPATH

cd /data/lzq/visualnav-transformer
python deployment/src/nomad_plan_service.py \
  _params_file:=deployment/config/nomad_service_params_goal_multisample.yaml
```

只靠 `conda activate rl` 还不够；缺少 ROS 的 `setup.bash` 和 catkin 的 `devel/setup.bash` 时，`rospy`、`cv_bridge`、`sensor_msgs`、`nomad_planner_msgs` 都不会完整可用。

## 10. 当前仓库默认配置提醒

当前 `src/user_config/user_config.yaml` 里默认是：

- `robot1_global_planner: theta_star`
- `robot2_global_planner: theta_star`

如果你要让 `robot2` 使用 NoMaD，需要显式改成：

```yaml
robot2_global_planner: diffusion
```

改完后重启 `scripts/main.sh`。

## 11. 常见问题

### 11.1 `No module named rospy`

你先进了 Conda 环境，但没先 source ROS：

```bash
source /opt/ros/noetic/setup.bash
source /data/lzq/ros_motion_planning/devel/setup.bash
```

然后再激活需要的 Conda 环境。

### 11.2 `No module named nomad_planner_msgs`

说明 catkin 工作区还没编译好，或者没 source：

```bash
cd /data/lzq/ros_motion_planning/scripts
./build.sh

cd /data/lzq/ros_motion_planning
source devel/setup.bash
```

### 11.3 `Timed out waiting for NoMaD service`

通常是 `move_base` 先起来了，但 `nomad_plan_service.py` 没起来，或者服务名不一致。

先查：

```bash
rosservice list | grep nomad
```

### 11.4 本地 `rsl_rl` 没生效

这一项只针对 PPO 路线。

训练或评估启动时，应看到类似日志：

```text
Using local rsl_rl from: /data/lzq/ros_motion_planning/src/rl_training/third_party/rsl_rl
```

如果没有，先确认脚本是从当前仓库运行的。

## 12. 最短可用路径

如果你的目标只是“在另一台机器装好这个修改版项目并能训练 TD3”，最短步骤是：

```bash
cd /data/lzq/ros_motion_planning

source /opt/ros/noetic/setup.bash
cd scripts
./build.sh

cd /data/lzq/ros_motion_planning
conda env create -f rl_environment.yaml
conda activate rl

python src/rl_training/train_velodyne_td3.py \
  --config src/rl_training/config/forklift_td3_noadv_nocurriculum.yaml
```

如果你的目标是“装好并训练 PPO”，最短步骤是：

```bash
cd /data/lzq/ros_motion_planning

source /opt/ros/noetic/setup.bash
cd scripts
./build.sh

cd /data/lzq/ros_motion_planning
conda env create -f rl_environment.yaml
conda activate rl

python src/rl_training/train_new.py \
  --config src/rl_training/config/forklift_ppo.yaml
```

如果你的目标是“跑扩散规划”，再额外按 `docs/nomad_movebase_setup.md` 配 `visualnav-transformer` 即可。
