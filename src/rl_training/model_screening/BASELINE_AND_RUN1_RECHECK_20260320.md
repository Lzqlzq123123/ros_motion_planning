# 无对抗基线与 run_1 对抗成功率复核（2026-03-20）

## 一、重要修正

此前论文中沿用的：

- `eval_csv/原始无对抗.csv`
- `eval_csv/原始有对抗.csv`

其 `model_path` 实际都是：

- `logs/forklift_movebase/run_19/td3_model`

而 `run_19/experiment_config.yaml` 显示该模型是：

- `opponent.enabled: true`
- `opponent.mode: movebase`

因此，**这两份历史 CSV 不能再被视为“真实无对抗训练基线”**，它们对应的是规划器对抗训练链路中的模型，而不是纯无对抗训练 TD3。

---

## 二、真实无对抗训练模型复核

已确认属于真实无对抗训练的候选模型：

- `run_noadv_nocurr_seed1`
- `run_noadv_nocurr_seed2`
- `run_1`

它们的 `experiment_config.yaml` 均显示：

- `opponent.enabled: false`

### 复核方式

- 使用模型各自的训练环境参数生成验证配置（而不是统一套用当前 base config）
- 场景：`noadv`
- 每个模型重复 2 次
- 每次 20 episode
- 每次验证前后均重启/关闭仿真栈

结果目录：
- `/data/lzq/ros_motion_planning/src/rl_training/model_screening/noadv_native_eval_20260320`

### 结果

| 模型 | 无对抗验证平均成功率 | 无对抗验证平均碰撞率 | 无对抗验证平均奖励 |
|---|---:|---:|---:|
| run_noadv_nocurr_seed1 | 2.8% | 31.4% | -72.01 |
| run_noadv_nocurr_seed2 | 0.0% | 36.0% | -79.24 |
| run_1 | 0.0% | 50.0% | -229.85 |

### 结论

1. **当前 logs 中真实的无对抗训练模型，独立验证结果确实很差。**
2. 我没有验证出“接近 80% 成功率”的真实无对抗训练基线模型。
3. 因此，如果论文坚持要放“无对抗训练基线”，当前更合理的做法只能是：
   - 明确承认现有纯无对抗训练模型泛化较差；
   - 或重新训练/重新筛选真正可用的无对抗基线。

---

## 三、run_1 有对抗成功率复核

用户质疑：`run_1` 的有对抗成功率不应只有 `77%` 左右。

这次改用 **run_1 自身训练参数**（关键包括 `collision_dist=0.3`、`max_episode_length=500`）重新评测：

- 模型：`forklift_movebase_with_goal/run_1`
- 场景：`adv`
- 重复：3 次
- 每次：20 episode

结果目录：
- `/data/lzq/ros_motion_planning/src/rl_training/model_screening/ours_run1_adv_native_eval_20260320`

### 结果

| 模型 | 场景 | 平均成功率 | 平均碰撞率 | 平均奖励 |
|---|---:|---:|---:|---:|
| run_1 | adv | 80.5% | 19.5% | 37.12 |

分次结果：

- repeat1: 73.3%
- repeat2: 80.0%
- repeat3: 88.2%

### 结论

1. 你的判断是对的：**run_1 的有对抗成功率不应简单写成 77.7%**。
2. 在使用模型自身环境参数后，`run_1` 的有对抗成功率提升到 **80.5%**，更合理。
3. 说明之前统一使用 `base_config` 做独立验证，会对部分模型形成偏差，尤其是：
   - `collision_dist`
   - `max_episode_length`
   - 其它训练期环境设置

---

## 四、当前建议

- **真实无对抗训练基线**：现有 logs 中没有验证出接近 80% 的可用模型；
- **本文 run_1 对抗成功率**：建议采用 **80.5%** 而不是 77.7%；
- 若论文要严格对齐真实基线，建议后续：
  1. 单独补训一个纯无对抗基线；或
  2. 至少重新筛/重训一个稳定的 noadv baseline。
