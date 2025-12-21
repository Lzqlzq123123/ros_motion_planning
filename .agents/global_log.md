# 全局日志

## 项目概述
- 项目名称: ros_motion_planning
- 当前任务: 集成 NoMaD 模型作为全局规划算法
- 创建时间: 2025-12-21
- 更新时间: 2025-12-21

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
  1. 加载 NoMaD 模型检查点
  2. 读取目标图像
  3. 获取当前相机观察
  4. 使用模型预测导航动作
  5. 转换为 ROS 路径消息

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
- **Context**: 插件通过~/<name>/service_name与goal_image_name参数配置服务名及目标图像，makePlan调用/nomad/make_plan并返回nav_msgs/Path供局部规划器消费；等待服务超时5秒并以持久连接方式复用客户端