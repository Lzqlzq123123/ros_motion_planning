# RSL RL 集成说明

本项目集成了自定义修改的 `rsl_rl` 库，用于 PPO 训练路线。

如果你只使用 TD3 训练或评估，可以直接跳过这份说明，不需要额外安装或配置 `rsl_rl`。

## 项目结构

```
ros_motion_planning/
├── src/rl_training/           # 训练脚本和配置
│   ├── train_new.py           # 训练脚本
│   ├── eval_trained_policy.py # 评估脚本
│   └── config/               # 配置文件
└── src/rl_training/
    ├── third_party/rsl_rl/          # 自定义修改的rsl_rl库
    │   └── rsl_rl/
    │       └── modules/
    │           └── res_actor_critic.py  # 新增的ResActorCritic网络
    └── config/
        └── forklift_ppo.yaml
```

## 修改内容

我们在 rsl_rl 库中添加了一个新的网络架构：

- **ResActorCritic**: 基于残差块的Actor-Critic网络，适用于机器人导航任务
- 位置: `src/rl_training/third_party/rsl_rl/rsl_rl/modules/res_actor_critic.py`

## 使用方法

### 1. 环境设置

首先创建conda环境并安装依赖：

```bash
conda env create -f rl_environment.yaml
conda activate rl
```

### 2. 使用自定义rsl_rl

项目已经配置为自动使用本地修改的rsl_rl库。训练和评估脚本会：

1. 自动检测本地rsl_rl库位置 (`src/rl_training/third_party/rsl_rl`)
2. 将其添加到Python路径
3. 优先使用本地版本而非pip安装的版本

### 3. 训练 PPO 模型

```bash
python src/rl_training/train_new.py --config src/rl_training/config/forklift_ppo.yaml
```

### 4. 评估 PPO 模型

```bash
python src/rl_training/eval_trained_policy.py \
    --config src/rl_training/config/forklift_ppo.yaml \
    --model src/rl_training/logs/forklift_ppo/run_1/model_100.pt \
    --num-episodes 5
```

## 配置说明

在 `src/rl_training/config/forklift_ppo.yaml` 中，我们已经配置使用新的ResActorCritic网络：

```yaml
runner:
  policy:
    class_name: "ResActorCritic"  # 使用自定义的残差Actor-Critic网络
    init_noise_std: 0.2
    # ... 其他配置
```

## 其他电脑上的设置

当其他电脑克隆此项目并使用 PPO 路线时，无需额外安装 `rsl_rl`，因为：

1. 本地rsl_rl库已经包含在项目中 (`src/rl_training/third_party/rsl_rl`)
2. 训练/评估脚本会自动使用本地版本
3. 只需要安装基础依赖即可（通过conda环境）

## 注意事项

- 请勿直接修改 `src/rl_training/third_party/rsl_rl` 中的Git历史
- 如需更新rsl_rl库，可以直接修改 `src/rl_training/third_party/rsl_rl` 中的文件
- 项目提交时会包含所有rsl_rl的修改，确保其他电脑获得相同版本
