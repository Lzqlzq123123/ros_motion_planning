# 全局日志

## 项目概述
- 项目名称: ros_motion_planning
- 当前任务: 修复 Forklift 环境中 LaserScan 初始化导致的异常奖励问题，并替换 Progress Reward 为 Distance Penalty
- 创建时间: 2025-12-21
- 更新时间: 2026-01-11

## 系统架构分析
- 当前实现了基于 move_base 的传统规划和基于 rsl_rl 的 gazebo 强化学习
- 支持通过 yaml 配置多个机器人
- 需要集成 /data/lzq/visualnav-transformer 中的 NoMaD 模型作为全局规划算法
- 目标: 在 yaml 中选择 diffusion 作为 global_planner,对应nomad模型

### 当前全局规划器实现
- 全局规划器通过 move_base 的 base_global_planner 参数配置
- 当前使用 path_planner/PathPlanner 作为插件
- 支持多种规划算法：astar, jps, dijkstra, dstar, rrt 等
- 配置文件位于 src/sim_env/config/planner/ 目录下

### NoMaD 模型分析
- 项目位置: /data/lzq/visualnav-transformer
- NoMaD (Navigation with Multimodal Affordances and Diffusion) 是一个视觉导航模型
- 主要组件：
  - train/vint_train/models/nomad/: 包含模型定义
  - deployment/: 包含部署脚本
  - deployment/goal/: 包含目标图像
- 输入：
  - 当前观察图像（相机图像）
  - 目标图像（goal image）
  - 机器人状态
- 输出：
  - 导航动作（速度命令或路径点）
- 推理流程：
  - 1. 加载 NoMaD 模型检查点
  - 2. 读取目标图像
  - 3. 获取当前相机观察
  - 4. 使用模型预测导航动作
  - 5. 转换为 ROS 路径消息

### 集成方案
- 创建一个 Python ROS 节点作为全局规划器
- 该节点订阅：
  - 相机图像话题
  - 机器人位姿话题
  - 目标点话题（用于选择对应的 goal image）
- 使用 NoMaD 模型进行路径规划
- 发布路径到 move_base 所需的话题

## 任务历史

### 初始分析 (2025-12-21)
- **Context**: 分析了 ros_motion_planning 项目结构和 NoMaD 模型架构，确定集成方案

### 集成方案确定 (2025-12-21)
- **分析结果**:
  - move_base 的全局规划器必须是 C++ 插件（继承 nav_core::BaseGlobalPlanner）
  - 通过 makePlan() 函数同步返回路径，而非话题发布
  - NoMaD 是 Python 实现，输出轨迹点序列
- **解决方案**:
  - C++ 轻量级包装器插件：调用 ROS 服务获取路径
  - Python NoMaD 服务节点：位于 /data/lzq/visualnav-transformer/deployment，提供规划服务
  - 局部规划器自动跟踪全局路径并计算控制量
- **架构**: 目标点 → move_base → C++ 插件 → ROS 服务 → Python NoMaD → 返回轨迹 → 局部规划器 → 速度命令

## TASK-001
- **Changes**: src/core/path_planner/nomad_planner/src/nomad_planner.cpp:1-182 -> 新增NoMaD全局规划器插件实现，封装ROS服务调用; src/core/path_planner/nomad_planner/include/nomad_planner/nomad_planner.h:1-49 -> 定义插件接口与参数; src/core/path_planner/nomad_planner/CMakeLists.txt:1-49 -> 配置库构建与安装; src/core/path_planner/nomad_planner/package.xml:1-31 -> 声明catkin依赖与插件导出; src/core/path_planner/nomad_planner/nomad_planner_plugin.xml:1-7 -> 注册nav_core插件
- **Line Stats**: +318, -1
- **Errors**: 初次缺少catkin_make命令；安装catkin后运行catkin_make -DCATKIN_WHITELIST_PACKAGES="" 成功编译
- **Context**: 插件通过~/[name]/service_name与goal_image_name参数配置服务名及目标图像，makePlan调用/nomad/make_plan并返回nav_msgs/Path供局部规划器消费；等待服务超时5秒并以持久连接方式复用客户端

## TASK-002
- **Changes**: /data/lzq/visualnav-transformer/deployment/src/nomad_plan_service.py:1-322 -> 新建NoMaD规划ROS服务节点，订阅相机并暴露/nomad/make_plan接口
- **Line Stats**: +322, -0
- **Errors**: 依赖rospy、torch等运行时未在当前环境验证；未实际跑通服务调用
- **Context**: 节点参数支持config_path、weights_path、goal_image_dir、goal_image_map与camera_topic；服务按请求goal_image_name解析目标图并运行扩散推理输出nav_msgs/Path，路径与输入frame一致并包含中间朝向

## TASK-002
- **Changes**: /data/lzq/visualnav-transformer/deployment/src/nomad_plan_service.py:1-330 -> 新增模型目录参数，支持通过model_name选择配置并归一化相对路径; /data/lzq/visualnav-transformer/deployment/config/models.yaml:1-16 -> 更新nomad权重路径至train/logs/nomad/49.pth
- **Line Stats**: +54, -7
- **Errors**: 未对cv2/torch依赖进行静态检查；需在部署环境验证模型目录路径是否存在；运行python部署脚本时报ModuleNotFoundError: nomad_planner_msgs，需在ros_motion_planning工作区source后提供PYTHONPATH；修正服务签名时出现AttributeError: MakePlan无Request属性，已改为显式导入MakePlanRequest/Response；rosservice调用返回Insufficient observations for planning，需先订阅camera并累计context_size张图像
- **Context**: 新增参数models_catalog_path与model_name，若未显式传入config/weights则从catalog解析，并默认指向0.jpg；路径归一化基于catalog所在目录与VISUALNAV_ROOT，便于在move_base launch中仅指定model_name=nomad

## TASK-003
- **Changes**: src/sim_env/launch/include/navigation/move_base.launch.xml -> 新增diffusion选项判断，选择nomad_planner/NoMaDPlanner并配置服务参数，附带启动Python服务节点; src/sim_env/scripts/nomad_plan_service_launcher.py -> 引入VISUALNAV_ROOT/NOMAD_PLAN_SERVICE_PATH环境变量执行真实服务脚本; src/sim_env/scripts/diffusion_policy/nomad_plan_service_launcher.py -> 退化为兼容包装器以调用上层脚本
- **Line Stats**: +91, -61
- **Errors**: 未运行roslaunch验证服务进程能否成功拉起；需在实际相机话题可用时联合测试路径规划流程
- **Context**: 当global_planner=diffusion时自动加载NoMaD全局规划器并以参数server_name/model_name配置服务，默认VISUALNAV_ROOT取自环境变量或/data/lzq/visualnav-transformer

## TASK-003 (auto)
- **Changes**: /data/lzq/visualnav-transformer/deployment/src/nomad_plan_service.py -> 新增goal目录轮询、自动规划、全局路径话题发布与相机投影可视化；src/sim_env/launch/include/navigation/move_base.launch.xml -> 增补NoMaD相机/位姿话题与plan topic参数以启用自动规划
- **Line Stats**: +290, -18
- **Errors**: 未在真实环境验证自动规划触发与路径跟随；需准备相机信息与机器人位姿话题，否则可视化回退到简化显示
- **Context**: 服务节点检测goal图像即触发NoMaD推理，输出路径到/$(robot_ns)/move_base/PathPlanner/plan并发布叠加图像plan_overlay，保持原有服务接口兼容；默认监控deployment/goal目录，处理后记录避免重复

## TASK-003 (env)
- **Changes**: src/sim_env/scripts/nomad_plan_service_launcher.py, src/sim_env/scripts/diffusion_policy/nomad_plan_service_launcher.py -> 默认回退到当前执行python，并允许通过NOMAD_PLAN_PYTHON指定自定义解释器；避免强绑/usr/bin/python3
- **Line Stats**: +28, -2
- **Errors**: 需保证外部设置NOMAD_PLAN_PYTHON时目标解释器包含torch等依赖；若保持默认conda环境需确保numpy版本兼容cv_bridge
- **Context**: 启动脚本优先使用已有rl环境，可根据需要设置NOMAD_PLAN_PYTHON切换解释器，从而平衡cv_bridge兼容性与torch可用性

## TASK-101
- **Changes**: docs/nomad_decoupled_design.md:1-129 -> 新增NoMaD解耦架构设计文档并在后续更新默认推理环境为nomad_train
- **Line Stats**: +129, -0
- **Errors**: 无
- **Context**: 文档记录外部服务交互、参数约定与失败恢复策略，为后续实现提供设计依据

## TASK-102
- **Changes**: src/core/path_planner/nomad_planner/include/nomad_planner/nomad_planner.h:1-74 -> 增加服务调用重试/超时配置与辅助函数声明; src/core/path_planner/nomad_planner/src/nomad_planner.cpp:1-210 -> 实现重试逻辑、请求超时日志与可选摘要输出
- **Line Stats**: +86, -9
- **Errors**: 无
- **Context**: 全局规划代理现在支持request_timeout、retry_count、log_plan_summary参数，提升与外部服务交互的鲁棒性

## TASK-103
- **Changes**: visualnav-transformer/deployment/launch/nomad_service.launch:1-20 -> 创建独立Launch文件，支持conda环境前缀、路径环境变量与透传额外参数
- **Line Stats**: +20, -0
- **Errors**: 无
- **Context**: Launch文件依赖sim_env包的启动脚本，默认使用nomad_train环境，可通过extra_args向服务脚本传参

## TASK-104
- **Changes**: src/sim_env/launch/include/navigation/move_base.launch.xml:16-78 -> 移除嵌入式Python服务节点，新增NoMaDPlanner的request_timeout/retry/log参数并精简所需参数集
- **Line Stats**: +6, -32
- **Errors**: 无
- **Context**: move_base配置现仅依赖外部服务，避免catkin内强制加载visualnav-transformer，保持局部规划链路不变

## TASK-105
- **Changes**: docs/nomad_decoupled_setup.md:1-65 -> 编写独立部署与联调指南，说明conda环境选择、启动顺序与常见问题
- **Line Stats**: +65, -0
- **Errors**: 无
- **Context**: 文档指导如何在nomad_train环境运行推理服务并与move_base联调，列出故障排查要点

## UPDATE-2025-12-22A
- **Changes**: visualnav-transformer/deployment/launch/nomad_service.launch:1-20 -> 删除依赖 sim_env 的 launch 文件；docs/nomad_decoupled_design.md:1-136 -> 更新推理端运行步骤与任务映射；docs/nomad_decoupled_setup.md:1-77 -> 改写启动指引为直接运行 Python 脚本并补充环境准备；backlog.json:1-35 -> 调整 TASK-102/103 描述与测试命令
- **Line Stats**: +36, -69
- **Errors**: 无
- **Context**: 推理服务改为通过手动执行 nomad_plan_service.py，在激活 nomad_train 环境并 source catkin 工作区后运行，移除对 roslaunch 与 sim_env 脚本的依赖；文档和 backlog 已同步说明新版流程

## UPDATE-2025-12-22B
- **Changes**: visualnav-transformer/deployment/config/models.yaml -> 新增 diffusion 模型条目供 move_base 使用；visualnav-transformer/deployment/config/nomad_service_params.yaml -> 新建统一参数配置；visualnav-transformer/deployment/src/nomad_plan_service.py -> 加载 YAML 默认参数并提供 ~params_file 覆盖逻辑；.agents/global_log.md -> 记录本次调整
- **Line Stats**: +84, -10
- **Errors**: 无
- **Context**: NoMaD 服务的所有参数改由 deployment/config/nomad_service_params.yaml 提供默认值，保持 ROS 参数覆盖能力并精简初始化逻辑；新增 diffusion 键以匹配全局规划器命名

## UPDATE-2025-12-22C
- **Changes**: visualnav-transformer/deployment/launch/nomad_service.launch -> 删除遗留的自动拉起 launch；ros_motion_planning/src/sim_env/scripts/nomad_plan_service_launcher.py 及 diffusion_policy/nomad_plan_service_launcher.py -> 移除旧版脚本；ros_motion_planning/README.md -> 新增手动启动 NoMaD 服务的步骤
- **Line Stats**: +42, -120
- **Errors**: 无
- **Context**: 清理错误技术路线产生的冗余脚本与 launch 文件，明确 NoMaD 仅通过命令行手动启动，并在 README 中描述从激活环境到运行 Python 服务的完整流程

## UPDATE-2025-12-27A
- **Changes**: src/rl_training/envs/ros_gazebo_env.py -> 在 PPO 环境重置时同步重置 robot2，并自动向其 goal 话题发布以 robot1 初始位置为终点的 PoseStamped；src/rl_training/config/forklift_ppo.yaml -> 新增 opponent 配置块以启用干扰机器人（model_name/goal_topic/frame_id）。
- **Line Stats**: +74, -0
- **Errors**: 未在真实 Gazebo+move_base 环境验证 robot2 自动导航链路，需要确保 robot2 move_base_simple/goal 订阅可用且控制器运行。
- **Context**: 便于在 PPO 训练 robot1 时使用 robot2 作为动态干扰体，自动随每次 robot1 reset 重置并下发目标，无需手动在 RViz 设置 goal。

## UPDATE-2025-12-29A
- **Changes**:
  - ros_gazebo_env.py: 对 robot2 发布的对抗目标增加固定 0.2m Y 方向偏移（map 帧）。目的：目标点不与 robot1 质心完全重合，避免 costmap 判定“当前位置即目标/碰撞”而直接返回 GOAL Reached 或无法规划。
  - pid_controller.cpp: 在 setPlan 中不再依赖目标是否变化，始终重置 `goal_x_/goal_y_/goal_theta_`、`goal_reached_` 以及积分项。目的：跨 episode/多次 reset 时即便目标坐标相同，也会重新进入跟踪，不再出现首个自动目标需要手动“激活”才能执行的情况。
- **Line Stats**: +32, -18（Python+CPP 总计）
- **Errors/Notes**: Python 改动无需编译；PID 改动需重新编译 catkin 后重启 move_base。偏移量可按需要调节（0.2m→0.3m）以适配场景。
- **Context/Impact**: 针对对抗场景（robot2 追撞 robot1）出现的“规划已完成但车不动/需手动下发一次 goal”问题。现在每次 reset 下发的自动目标都会被视为新目标并执行；同时减少因 footprint 重叠导致的规划失败。

## TASK-202
- **Changes**: src/rl_training/eval_trained_policy.py:1-96 -> 引入 `get_user_config_init_poses` 函数，从 `src/user_config/user_config.yaml` 读取机器人初始位置，并注入到 `env_cfg` 中。
- **Line Stats**: +45, -0
- **Errors**: 无
- **Context**: 修复了 `eval_trained_policy.py` 运行时，由于缺少 `init_poses` 配置，导致所有机器人被重置到 (0,0) 原点并发生碰撞的问题。现在评估脚本会复用 `user_config.yaml` 中的初始位置配置，与训练脚本 `train_new.py` 行为一致。

## TASK-203
- **Changes**: src/rl_training/config/forklift_ppo.yaml:1-84 -> 新增 `action_scale: [0.5, 0.5]` 参数; src/rl_training/envs/ros_gazebo_env.py:1-476 -> 在 `set_action` 中应用缩放，将 PPO 输出 [-1, 1] 映射到物理速度。
- **Line Stats**: +29, -10
- **Errors**: 无
- **Context**: 修复了训练时机器人“原地打转”的问题。原因为 PPO 输出直接被 clip 到 [-0.5, 0.5]，导致大动作被截断且梯度消失。引入缩放后，动作空间映射更合理。

## TASK-204
- **Changes**: src/rl_training/config/forklift_ppo.yaml:1-85 -> 新增 `control_dt: 0.1` 参数; src/rl_training/envs/ros_gazebo_env.py:1-480 -> 修改 `step` 和 `reset` 方法，使用 `/gazebo/pause_physics` 和 `/gazebo/unpause_physics` 服务配合 `rospy.sleep(control_dt)` 实现离散步进控制。
- **Line Stats**: +45, -15
- **Errors**: 无
- **Context**: 替换了原有的 `rospy.sleep(0.1)`，通过显式控制物理引擎的暂停和恢复，确保每个训练步长的物理时间精确为 `control_dt`，解决了训练时序不稳定的问题。同时在 reset 时也加入了 unpause/pause 逻辑以确保状态正确更新。

## TASK-205 (Analysis)
- **Problem**: 当前奖励函数存在严重的 "Reward Farming" 漏洞。
  - `at_goal_pos` 判定条件（距离 < 0.05m）会每步触发 `goal_reward * 0.5` (10.0)，即使机器人没有停下或朝向不对。这导致机器人倾向于在目标点附近震荡以刷取奖励，而不是完成任务。
  - 存在冗余的距离奖励：`displacement_reward` (投影距离) 和 `progress_reward` (距离差值) 同时存在且系数均为 1.0，导致趋向目标的奖励权重过大。
- **Plan**:
  1. **Fix Goal Reward**: 将 Goal Reward 改为稀疏的 Terminal Reward。只有在 `done=True` 且满足所有成功条件（位置+朝向+静止）时才给予一次性奖励。
  2. **Remove Redundant Reward**: 移除 `displacement_reward` (投影法)，保留 `progress_reward` (势能差法)，因为后者数学性质更好（保证总奖励 = 总距离 * scale）。
  3. **Cleanup**: 移除 `at_goal_pos` 的每步奖励逻辑。

## TASK-206 (New Issue)
- **Problem**: 用户报告奖励值异常大（负值），怀疑是 Scan 初始化为 0 导致误报碰撞。
- **Analysis**: 代码中 `self.scan = np.zeros(self.scan_dim)`。在 `scan_cb` 第一次回调前，`min_scan` 为 0。
- **Impact**: `compute_reward_and_done` 中检测到 `min_scan < collision_dist` (0 < 0.18)，触发 `collision_penalty` (-10.0) 和 `min_range_penalty` (-2.0)。导致初始 steps 即使无碰撞也产生 -12.0 的 step reward。
- **Plan**: 将 `self.scan` 初始化为 `np.full(self.scan_dim, 10.0)` 或其他安全的大数值。

## TASK-206
- **Changes**: src/rl_training/envs/ros_gazebo_env.py:265 -> `self.scan` 初始化从 `np.zeros` 改为 `np.full(..., 100.0)`
- **Line Stats**: +2, -1
- **Errors**: 无
- **Context**: 修复了 LaserScan 在第一次回调前默认为 0，导致 `compute_reward_and_done` 误判为碰撞（min_scan < 0.18），从而在初始时刻给予巨大的负奖励问题。现在初始化为 100.0 (大于 max range)，确保在传感器数据到来前被视为无障碍。

## TASK-207
- **Changes**: 
  - src/rl_training/config/forklift_ppo.yaml: 新增 `dist_penalty_scale: -0.5`，注释掉 `dist_reward_scale`。
  - src/rl_training/envs/ros_gazebo_env.py: 移除 Progress Reward 计算，替换为 Distance Penalty (`dist_to_goal * dist_penalty_scale`)。
- **Line Stats**: +6, -5
- **Errors**: 无
- **Context**: 用户要求“距离目标越远惩罚越大”，因此移除了基于差值的 Progress Reward（只关注局部改进），改为基于绝对距离的 Distance Penalty（关注全局状态）。这有助于提供持续的负反馈，引导 Agent 尽快缩短到目标的距离。

## TASK-208
- **Changes**:
  - src/rl_training/config/forklift_ppo.yaml: `action_penalty_scale` -> 0.0, `heading_penalty_scale` -> -0.5, Removed `pose_gate_radius`, `yaw_reward_scale`, `success_stay_steps`.
  - src/rl_training/envs/ros_gazebo_env.py: Simplified reward logic. Removed `Milestone Reward`, `pose_gate_radius` logic. `at_goal_pos` now immediately triggers `done=True` and `goal_reward`.
- **Line Stats**: +20, -40 (Approx)
- **Errors**: None
- **Context**: Simplified reward function to fix oscillation and "reward farming". By removing conflicting pose/action penalties and complex multi-stage success criteria, the agent should focus on the primary objective: reaching the goal position.
