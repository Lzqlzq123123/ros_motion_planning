# Global Change Log

## 2026-03-23
- [修改] `CLAUDE.md` — 删除“每次修改都必须追加记录到 .agents/global_log.md”的规则

## 2026-03-19
- [修改] `src/rl_training/envs/movebase_gazebo_env.py` — 重构 goal/spawn 配置逻辑：新增统一的 `goal_range(x_min/x_max/y_min/y_max)`、`goal_mode(fixed/random)`、`ego_spawn_radius`、`adv_spawn_radius`、`goal_offset` 语义；课程学习改为在基础范围上按 `delta` 向四边扩展，累计上限为 `max_span`；删除旧的 `random_spawn` / 固定起点分支依赖
- [修改] `src/rl_training/eval_velodyne_td3_with_goal.py` — 评估脚本改为直接复用 env 内部 goal 采样逻辑，仅保留 fixed goal 或 goal_range 覆盖入口，删除旧的随机 goal 采样辅助逻辑与起点模式分支
- [修改] `src/rl_training/train_velodyne_td3_with_goal.py` — 训练脚本同步统一 goal 配置覆盖逻辑，删除旧的 `goal_random_range` / `start_pose_mode` 流程，避免 reset 后再次手动覆盖 goal
- [修改] `src/rl_training/config/forklift_movebase.yaml` — 配置项重构为 `goal_range`、`curriculum(delta/max_span)`、`ego_spawn_radius`、`adv_spawn_radius`、`goal_offset`
- [修改] `src/rl_training/config/forklift_td3_noadv_nocurriculum.yaml` — 同步切换到新的 goal/spawn 配置结构，并清理旧字段与脏内容
- [修改] `src/rl_training/train_velodyne_td3.py` — 新增 fixed-point / goal_range CLI 覆盖、对手模式与 goal_offset 覆盖，并将评估/实验记录切换到新的 `goal_mode`、`goal_range`、`curriculum_expansion` 语义
- [修改] `src/rl_training/eval_velodyne_td3.py` — 对齐新的 goal 配置覆盖逻辑，支持 fixed 单点与随机 goal_range 评估入口
- [错误] 重构过程中一度误删 `append_episode_result` / `if __name__ == "__main__"` 结构，已修复并通过 `python -m py_compile` 完成语法检查

## 2026-03-11
- [修改] `src/rl_training/envs/movebase_gazebo_env.py` — 修复一系列 AttributeError：
  - 新增 `MoveBaseGazeboEnv.publish_global_goal_marker()` 方法
  - 新增 `MoveBaseRobot.publish_goal_marker()` 辅助方法
  - `set_absolute_goal()` 新增 `publish_move_base` 参数
  - 新增 `MoveBaseGazeboEnv.robot1_mode` 属性从 env_cfg 读取
  - 新增 `opponent_mode="diffusion"` 支持：使用 NoMaD diffusion planner（通过 move_base 调用 `/nomad/make_plan`）

## 2026-03-10

### 2D Diffusion Ablation Study (无视觉编码器 baseline)

论文消融实验：对比 NoMaD（有视觉编码）vs 纯位置条件 diffusion（无视觉）

- [新增] `/data/lzq/visualnav-transformer/train/vint_train/models/diffusion_2d/diffusion_2d.py` — 无视觉编码器的 2D diffusion 模型：
  - `PositionEncoder`: 位置历史 → 潜在向量 (MLP: history_len*2 → 128 → 256)
  - `Diffusion2D`: 仅用位置条件生成轨迹，复用 NoMaD 的 `ConditionalUnet1D`
- [新增] `/data/lzq/visualnav-transformer/train/vint_train/data/position_dataset.py` — 纯位置数据集：
  - 加载 traj_data.pkl 中的 position/yaw，不加载图像
  - 输出：(positions_history, actions, distance)
- [新增] `/data/lzq/visualnav-transformer/train/train_diffusion_2d.py` — 训练脚本
- [新增] `/data/lzq/visualnav-transformer/train/config/diffusion_2d.yaml` — 配置文件
- [新增] `/data/lzq/visualnav-transformer/train/scripts/generate_gazebo_split.py` — 生成 gazebo 数据集的 train/test split
- [新增] `/data/lzq/visualnav-transformer/train/scripts/train_diffusion_2d.sh` — 启动脚本

**启动命令**:
```bash
conda activate rl
cd /data/lzq/visualnav-transformer/train
python train_diffusion_2d.py --config config/diffusion_2d.yaml
```

## 2026-03-07
- [新增] `src/rl_training/train_velodyne_td3.py` — 新增实验参数自动记录功能：
  - `get_git_info()`: 获取当前 git commit/branch/dirty 状态
  - `save_experiment_config()`: 保存完整实验配置到 `logs/<experiment>/<run>/experiment_config.yaml`
  - 记录内容包括：metadata（时间戳、git 信息）、environment（机器人、地图、碰撞阈值等）、opponent（是否对抗、模式、参数）、network（Actor/Critic 层结构）、hyperparameters（全部 TD3 超参数）、training（训练配置）、runner（实验名称、种子）

## 2026-02-27
- [修改] `CLAUDE.md` — 全面更新项目分析文档：新增全局规划算法清单（20+种）、局部控制器清单（9种）、TD3 训练管线详细分析（观测/动作空间、奖励设计、课程学习、探索策略、网络结构）、数据流架构图、当前配置说明、项目亮点总结；标注 PPO 算法已废弃，主要使用 TD3

## 2026-02-24
- [修改] `src/rl_training/envs/movebase_gazebo_env.py` — 新增课程学习：`_curriculum_sample_goal` 方法，goal_span 从 ±4m 逐步扩展到 ±10m，每次 reset 增加 0.004
- [修改] `src/rl_training/train_velodyne_td3.py` — TensorBoard 新增 `train/curriculum_span` 指标记录
- [修改] `src/rl_training/config/forklift_movebase.yaml` — 新增 `curriculum` 配置块（initial_span/max_span/delta）

## 2026-02-14

### 项目分析与文档建设
- [新增] `CLAUDE.md` — 创建项目指导文档，包含构建命令、架构说明、配置系统、RL 训练流程
- [修改] `CLAUDE.md` — 添加 conda `rl` 环境强制要求、Change Log 记录规则

### Diffusion 对抗规划器设计与初步实现

**背景**: 当前 robot2 使用 NoMaD（视觉 diffusion 模型）生成全局轨迹，存在三个问题：
1. NoMaD 基于相机图像 + goal image，在 Gazebo warehouse 中视觉语义不匹配，轨迹质量差
2. NoMaD 不感知 costmap，生成的轨迹穿墙、不避障
3. 对抗目标仅为 robot1 终点偏移 0.5m，不构成有效拦截

**方案决策**: 设计纯 2D costmap-conditioned diffusion 模型替代 NoMaD
- 训练数据: 80% 随机 A\* 轨迹 + 20% 对抗拦截轨迹（离线生成，无需 Gazebo）
- 推理时: dual-guided sampling（costmap 引导避障 + 对抗引导拦截 robot1）
- 保持 `nomad_planner` C++ 插件和 `MakePlan.srv` 接口不变
- 详细设计方案见 `.claude/plans/sunny-gathering-breeze.md`

**已完成文件**:
- [新增] `src/rl_training/diffusion_planner/__init__.py` — 包初始化
- [新增] `src/rl_training/diffusion_planner/astar.py` — 纯 Python A\* 路径规划 + OccupancyMap 地图加载（含膨胀、距离场、局部 patch 提取）
- [新增] `src/rl_training/diffusion_planner/data/` — 训练数据存放目录
- [新增] `src/rl_training/diffusion_planner/config/` — 配置文件目录

**数据方案调整**: 发现已有 rosbag 数据集 `/data/lzq/datasets/nav_diffusion/`（204 条真实导航轨迹，20251204 55 bags + 20260131 149 bags），改为直接从 rosbag 提取轨迹训练，不做数据增强。

**已完成全部实现文件**:
- [新增] `src/rl_training/diffusion_planner/extract_rosbag_data.py` — 从 rosbag 提取 (start, goal, trajectory)，重采样 K=16 waypoints，配对 costmap patch，保存 .npz
- [新增] `src/rl_training/diffusion_planner/model.py` — DDPM 模型（CostmapEncoder CNN + StateEncoder MLP + DenoisingMLP + cosine/linear schedule），~1.5M 参数
- [新增] `src/rl_training/diffusion_planner/dataset.py` — PyTorch Dataset，z-normalisation
- [新增] `src/rl_training/diffusion_planner/train_diffusion.py` — 训练脚本（Adam + cosine LR + TensorBoard + checkpoint）
- [新增] `src/rl_training/diffusion_planner/guidance.py` — Dual-guided sampling（costmap 排斥引导 + 对抗吸引引导），inference-time gradient guidance
- [新增] `src/rl_training/diffusion_planner/costmap_diffusion_service.py` — ROS 服务节点，提供 `/nomad/make_plan`，订阅 `/robot1/rl_goal` 用于对抗引导
- [新增] `src/rl_training/diffusion_planner/config/default.yaml` — 完整配置文件
- [修改] `src/rl_training/envs/movebase_gazebo_env.py` — RobotHandle 添加 `/robot1/rl_goal` PoseStamped publisher（2 行改动）

### visualnav-transformer 项目 (`/data/lzq/visualnav-transformer/`)

**现状**: NoMaD 扩散模型视觉导航服务
- `deployment/src/nomad_plan_service.py` — 提供 `/nomad/make_plan` ROS 服务
- 模型结构: EfficientNet-B0 视觉编码器 + ConditionalUnet1D 去噪网络 + 距离预测 MLP
- 输入: 相机图像序列 + goal image → 输出: 16 个相对 waypoints
- 配置: `train/config/nomad.yaml`（context_size=3, len_traj_pred=16, num_diffusion_iters=10）
- 训练数据归一化: ACTION_STATS min=[-10,-18] max=[18,18]

**计划**: 新的 costmap diffusion service 将替代 nomad_plan_service.py 的角色，但 nomad_plan_service.py 本身不删除（保留作为 baseline 对比实验用）

## 2026-02-22

### Diffusion 对抗轨迹选择改造（NoMaD unconditional 多轨迹采样 + 碰撞评分）

- [修改] `/data/lzq/visualnav-transformer/deployment/config/nomad_service_params.yaml` — 新增 adversarial trajectory selection 参数块（adversarial_selection_enabled, num_adversarial_samples=8, adversarial_collision_radius, adversarial_time_decay=0.9, ego_speed_estimate=0.3）
- [修改] `/data/lzq/visualnav-transformer/deployment/src/nomad_plan_service.py` — 多处改动：
  - `__init__`: 读取 5 个新对抗参数；订阅条件扩展为 `opponent_guidance_enabled or adversarial_selection_enabled`；启动日志输出对抗模式状态
  - `_run_nomad`: 对抗模式下 goal_mask=1（unconditional），obsgoal_cond repeat N 份并行去噪生成 N 条候选轨迹 (N,T,2)；非对抗模式保持原有单轨迹行为
  - 新增 `_predict_ego_trajectory`: 基于 robot1 odom+goal 线性外推 ego 未来 T 步世界坐标
  - 新增 `_select_adversarial_trajectory`: 将 N 条 robot frame 候选轨迹转 world frame，与 ego 预测路径逐点计算加权碰撞分数 Σ(γ^t / dist)，选 argmax
  - `_handle_make_plan`: 在 `_run_nomad` 返回后插入对抗选择，将 (N,T,2) 降为 (T,2) 后继续走 classical guidance 流程

### Opponent 随机生成改造

- [修改] `src/rl_training/config/forklift_movebase.yaml` — opponent 新增 `spawn_near_goal: true` 和 `spawn_radius: [1.0, 3.0]`
- [修改] `src/rl_training/envs/movebase_gazebo_env.py` — `reset_opponent()` 改为接收 goal 坐标，在 `[r_min, r_max]` 环形区域内随机采样无碰撞位置生成 robot2；调用顺序改为先 reset_opponent 再 publish_opponent_goal
- [修改] `src/rl_training/envs/movebase_gazebo_env.py` — 重写生成流程：删除 `change_goal` expanding window 逻辑，改为先随机生成 goal → robot1 在 goal 周围 `ego_spawn_radius` 距离处生成 → robot2 在 goal 周围 `opponent.spawn_radius` 距离处生成；提取公共方法 `_sample_free_pos` 和 `_sample_around`
- [修改] `src/rl_training/config/forklift_movebase.yaml` — 新增 `ego_spawn_radius: [3.0, 6.0]`，删除 opponent `init_pose`

### 全链路代码审查与修复

- [修复] `/data/lzq/visualnav-transformer/deployment/config/nomad_service_params.yaml` — `opponent_goal_topic` 从 `/robot1/move_base_simple/goal` 改为 `/robot1/rl_goal`，修复 topic 不通导致对抗选择完全失效的问题
- [修复] `src/rl_training/envs/movebase_gazebo_env.py` — `publish_opponent_goal` 移除 `goal_offset` 偏移，直接发送 robot1 的真实 goal 坐标
- [修复] `src/rl_training/envs/movebase_gazebo_env.py` — `reset_opponent` 前先发零速 cmd_vel 给 robot2 停车，避免 reset_world 后残留旧速度
- [删除] `src/rl_training/envs/movebase_gazebo_env.py` — 移除死代码 `MoveBaseRobot.reset_goal`
- [修改] `src/rl_training/config/forklift_movebase.yaml` — 删除不再使用的 `goal_offset` 字段
