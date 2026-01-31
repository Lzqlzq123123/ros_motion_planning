# NoMaD全局规划启动指南

## 概述

NoMaD (Goal Masking Diffusion Policies for Navigation and Exploration) 是一种基于扩散模型的视觉导航策略，可作为ROS运动规划系统的全局规划器。本指南将详细说明如何在ROS运动规划项目中启动和使用NoMaD进行全局路径规划。

## 系统架构

NoMaD在ROS运动规划系统中的集成架构如下：

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RViz/GUI      │    │   Move_Base      │    │   机器人控制    │
│   (目标设置)     │───▶│  (路径规划)      │───▶│   (速度命令)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ NoMaDPlanner     │
                       │ (全局规划器)      │
                       └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ NoMaD规划服务     │
                       │ (扩散模型推理)    │
                       └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ PID控制器         │
                       │ (局部规划器)      │
                       └──────────────────┘
```

## 前置条件

### 1. 环境要求

- Ubuntu 18.04/20.04
- ROS Noetic
- Python 3.7+
- CUDA 10+ (用于GPU加速)
- 依赖的Python包：
  - PyTorch
  - OpenCV
  - NumPy
  - PIL (Pillow)
  - diffusers
  - yaml

### 2. 模型文件准备

确保已下载NoMaD预训练模型，并将其放置在正确位置：

1. 从[官方链接](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)下载NoMaD模型权重
2. 将模型文件(`*.pth`)放置到`../visualnav-transformer/train/logs/nomad/`目录下
3. 确保配置文件`../visualnav-transformer/train/config/nomad.yaml`存在

### 3. 目标图像准备

NoMaD需要目标图像来生成导航路径：

1. 准备目标环境的目标图像(例如：仓库的特定区域)
2. 将图像放置到`../visualnav-transformer/deployment/goal/`目录下
3. 默认使用`goal.jpg`作为目标图像

## 配置文件详解

### 1. NoMaD服务参数配置

文件位置：`../visualnav-transformer/deployment/config/nomad_service_params.yaml`

```yaml
# NoMaD规划服务基本配置
service_name: /nomad/make_plan          # ROS服务名称
camera_topic: /robot2/camera/rgb/image_raw     # 相机图像话题
camera_info_topic: /robot2/camera/rgb/camera_info  # 相机参数话题
robot_pose_topic: /robot2/ground_truth/state     # 机器人位姿话题
plan_topic: /robot2/move_base/PathPlanner/plan  # 规划路径发布话题
plan_overlay_topic: plan_overlay          # 路径可视化图像话题

# 相机物理参数 (根据实际机器人调整)
camera_height: 0.95                     # 相机高度 (米)
camera_x_offset: 0.45                    # 相机x轴偏移 (米)

# 路径生成参数
waypoint_spacing: 5                      # 采样点间隔 (索引个数)
metric_waypoint_spacing: 0.12            # 实际距离间隔 (米)

# 目标图像设置
goal_image_dir: deployment/goal          # 目标图像存放路径
default_goal_image: goal.jpg             # 默认目标图像文件名

# 模型配置
models_catalog_path: deployment/config/models.yaml  # 模型目录文件
model_name: diffusion                    # 在models.yaml中定义的模型名
config_path: "train/config/nomad.yaml"   # 模型配置文件
weights_path: "train/logs/nomad/49.pth"  # 模型权重文件
```

### 2. 机器人配置

文件位置：`src/user_config/user_config.yaml`

```yaml
robots_config:
  - robot2_type: "forklift"              # 机器人类型
    robot2_global_planner: "diffusion"   # 使用NoMaD作为全局规划器
    robot2_local_planner: "pid"          # 使用PID作为局部规划器
    robot2_x_pos: "2.0"                  # 初始X位置
    robot2_y_pos: "0.0"                  # 初始Y位置
    robot2_z_pos: "0.0"                  # 初始Z位置
    robot2_yaw: "0.0"                    # 初始偏航角
```

## 启动步骤

### 1. 设置环境变量

```bash
# 设置visualnav-transformer根目录路径
export VISUALNAV_ROOT=/data/lzq/visualnav-transformer

# 确保Python路径包含所需模块
export PYTHONPATH=$VISUALNAV_ROOT:$VISUALNAV_ROOT/train:$VISUALNAV_ROOT/deployment/src:$VISUALNAV_ROOT/diffusion_policy:$PYTHONPATH
```

### 2. 启动NoMaD规划服务

```bash
# 方法1：使用Python直接启动
cd /data/lzq/visualnav-transformer
rosrun deployment nomad_plan_service.py _params_file:=deployment/config/nomad_service_params.yaml

# 方法2：使用roslaunch启动(推荐)
roslaunch sim_env nomad_service.launch
```

### 3. 启动仿真环境

```bash
# 启动Gazebo仿真和机器人
cd /data/lzq/ros_motion_planning
roslaunch sim_env config.launch \
  world:=warehouse \
  map:=warehouse \
  robot_number:=2 \
  rviz_file:=sim_env.rviz
```

### 4. 验证系统运行

```bash
# 检查NoMaD服务是否正常运行
rosservice call /nomad/make_plan "start:
  header:
    seq: 0
    stamp: {secs: 0, nsecs: 0}
    frame_id: 'map'
  pose:
    position: {x: 2.0, y: 0.0, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
goal:
  header:
    seq: 0
    stamp: {secs: 0, nsecs: 0}
    frame_id: 'map'
  pose:
    position: {x: 5.0, y: 3.0, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
goal_image_name: 'goal.jpg'"
```

### 5. 设置导航目标

使用RViz设置导航目标：

1. 启动RViz
2. 使用"2D Nav Goal"工具在地图上设置目标点
3. 观察NoMaD生成的路径和机器人的运动

## 常见问题与解决方案

### 1. 模型加载失败

**问题**：`FileNotFoundError: NoMaD weights file not found`

**解决方案**：
- 检查`weights_path`是否正确指向模型文件
- 确保模型文件已下载并放置在正确位置
- 验证文件权限是否允许读取

### 2. 相机图像订阅失败

**问题**：无法获取相机图像

**解决方案**：
- 检查`camera_topic`是否与实际相机话题匹配
- 确认相机节点已正确启动
- 使用`rostopic echo`验证话题是否存在

### 3. 规划服务连接超时

**问题**：`Timed out waiting for NoMaD service`

**解决方案**：
- 确保NoMaD规划服务已先启动
- 检查`service_name`配置是否正确
- 增加`connect_timeout`参数值

### 4. 路径规划失败

**问题**：NoMaD返回空路径或规划失败

**解决方案**：
- 检查目标图像是否清晰且与目标位置匹配
- 确认机器人位置估计是否准确
- 调整`waypoint_spacing`和`metric_waypoint_spacing`参数

### 5. 性能优化

**建议**：
- 使用GPU加速：确保CUDA环境正确配置
- 调整`num_diffusion_iters`参数(默认50，可减少以提高速度)
- 减小图像尺寸：修改`image_size`参数

## 高级配置

### 1. 自定义模型配置

如果使用自定义训练的NoMaD模型，需要修改以下文件：

1. 更新`../visualnav-transformer/deployment/config/models.yaml`，添加新模型条目：

```yaml
my_custom_nomad:
  config_path: "train/config/my_nomad.yaml"
  ckpt_path: "../model_weights/my_nomad.pth"
```

2. 在`nomad_service_params.yaml`中指定`model_name`为`my_custom_nomad`

### 2. 多机器人配置

对于多机器人系统，每个机器人需要独立的NoMaD服务实例：

```bash
# 为robot2启动NoMaD服务
rosrun deployment nomad_plan_service.py _params_file:=deployment/config/nomad_service_params.yaml _service_name:=/robot2/nomad/make_plan

# 更新robot2的move_base配置
roslaunch sim_env config.launch robot_number:=2
```

### 3. 动态目标图像切换

可以在运行时切换目标图像：

```bash
# 发布新的目标图像
rosservice call /nomad/make_plan "start: {...} goal: {...} goal_image_name: 'new_target.jpg'"
```

## 调试与监控

### 1. 日志查看

```bash
# 查看NoMaD服务日志
rosnode info /nomad_plan_service

# 启用详细日志输出
rosservice call /nomad/make_plan "..." --verbose
```

### 2. 可视化工具

- **路径可视化**：RViz中的Path显示
- **相机图像**：RViz中的Image显示
- **规划叠加**：`plan_overlay_topic`发布的图像显示路径在相机图像上的投影

### 3. 性能监控

```bash
# 监控服务调用频率和响应时间
rostopic hz /robot2/move_base/PathPlanner/plan

# 监控CPU/GPU使用率
nvidia-smi  # 对于GPU加速
htop        # 对于CPU使用
```

## 结论

NoMaD为ROS运动规划系统提供了强大的视觉导航能力，特别适合于动态环境下的复杂导航任务。通过本指南，您应该能够成功配置和启动NoMaD作为全局规划器，并在仿真环境中进行测试。

如需更多技术细节或遇到问题，请参考：
1. [NoMaD原论文](https://general-navigation-models.github.io/nomad/index.html)
2. [VisualNav Transformer项目文档](https://github.com/robodhruv/visualnav-transformer)
3. ROS Move Base文档