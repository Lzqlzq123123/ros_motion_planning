# NoMaD 推理与 move_base 解耦设计

## 目标
- 将 NoMaD 视觉推理流程隔离在独立 Conda 环境与工作区内，避免与 catkin 依赖冲突
- 继续复用 ros_motion_planning 的 move_base 导航栈（尤其是局部规划与控制输出）
- 通过标准 ROS 接口交互，让任意全局规划触发逻辑都可以调用 NoMaD 生成的全局路径

## 总体架构
```
+----------------+         ROS Service          +-------------------------+
| move_base 节点 | <--------------------------> | DiffusionProxy Planner  |
|  (catkin ws)   |    /nomad/make_plan 请求     | (nav_core::BaseGlobal)  |
+--------+-------+                               +------------+------------+
         ^                                                        |
         | Path                                                    | Service call
         |                                                         v
         |                                           +---------------------------+
         |                                           | nomad_plan_service.py     |
         |                                           | (visualnav Conda env)     |
         |                                           +------------+--------------+
         |                                                        |
         |                                           Camera / Pose Topics
         +---------------------------------------------------------+
```

### 关键组件
1. **DiffusionProxy Planner**（C++ 或 Python 包装均可）
   - 运行在 ros_motion_planning catkin 工作区
   - 继承 `nav_core::BaseGlobalPlanner`
   - `makePlan` 内部通过 ROS service client 调用 `/nomad/make_plan`
   - 将 service 返回的 `nav_msgs/Path` 原样交给 move_base
   - 不直接依赖 Torch / cv_bridge

2. **nomad_plan_service.py**（推理服务）
  - 运行在 visualnav-transformer 仓库，使用 `conda envs/nomad_train` 或独立环境
   - 订阅传感器话题（图像、tf、定位）并执行 NoMaD 模型推理
   - 提供 `/nomad/make_plan` 服务接口：根据输入初始位姿 / 目标图像生成全局轨迹
   - 可以额外发布调试可视化话题（路径、覆盖图像）

3. **运行与配置**
  - 推理端：用户在 visualnav-transformer 工作区手动运行 `nomad_plan_service.py`（例如 `conda run -n nomad_train python deployment/src/nomad_plan_service.py --service_name /nomad/make_plan`），确保已 `source` 对应 catkin 工作区以加载消息定义。
  - 导航端：`ros_motion_planning/src/sim_env/launch/move_base.launch.xml` 设置 `base_global_planner` 为代理插件，并通过参数配置服务名称、超时与重试策略；必要时可叠加额外监控节点处理故障切换。

## 数据流
1. 上层触发导航（例如 RViz 下发 goal 或自动模式请求）：
   - move_base action server 接到目标
   - move_base 请求全局规划 → 调用 Proxy Planner 的 `makePlan`
2. Proxy Planner：
   - 构造 `/nomad/make_plan` 请求，包含起点 pose、目标描述（goal pose 或图像引用）
   - 等待服务响应，超时则返回失败给 move_base（可配置重试次数）
3. NoMaD 推理服务：
   - 使用传感器缓存与模型推理生成 `nav_msgs/Path`
   - 返回路径并根据需要发布可视化话题
4. move_base：
   - 收到有效路径后更新全局规划
   - 局部规划器继续使用现有扫描/里程计话题输出速度指令

## 接口规范
- **Service** `/nomad/make_plan`
  - Request：
    - `geometry_msgs/PoseStamped start`
    - `nomad_msgs/ImageGoal goal`（自定义消息，包含图像指针或话题名）
    - `float32 timeout`
  - Response：
    - `nav_msgs/Path path`
    - `bool success`
    - `string message`
- **参数**（通过 ROS 参数服务器）：
  - `/nomad/service_name`：默认 `/nomad/make_plan`
  - `/nomad/request_timeout`：默认 3.0 s
  - `/nomad/auto_trigger`：是否允许服务端在接收目标图像后主动推送路径（可选）

## 部署与环境隔离
- **推理服务**
  - 启动前 `source /opt/ros/noetic/setup.bash` 以及 catkin 工作区 `devel/setup.bash`
  - `conda run -n nomad_train python /data/lzq/visualnav-transformer/deployment/src/nomad_plan_service.py --service_name /nomad/make_plan`（或在激活环境后直接调用 `python ...`）
- **导航栈**
  - 使用系统 Python / ROS Noetic 环境
  - catkin 编译时仅依赖标准 ROS 包和自定义消息

## 失败与恢复策略
- Proxy Planner 调用服务失败时：
  - 记录 ROS_WARN 并返回 false，让 move_base 执行 fallback（比如原生 Dijkstra planner）
  - 可选：在参数中提供 `fallback_planner` 字段，允许自动切换
- 服务端推理异常：
  - 返回 `success=false` 与错误信息供 Proxy 记录
  - 可通过 diagnostics 或 heartbeat 话题暴露服务健康状况

## 开放问题
- 服务接口中 goal 的图像引用方式（直接传入数据 vs. 仅传 topic 名称）需要与现有 NoMaD 逻辑对齐
- cv_bridge 兼容性：若服务端升级到支持 NumPy 2 的自编译版本，需要在 launch 文档中说明构建步骤
- 多机器人场景下的命名空间隔离：service 名称、传感器话题是否需加 robot 命名空间（建议在 launch 中支持 `<group ns="robotX">`）

## 后续任务映射
- `TASK-102`：实现 Proxy Planner
- `TASK-103`：提供推理服务独立运行指引
- `TASK-104`：调整 move_base 配置参数
- `TASK-105`：产出运维文档与调试指南
