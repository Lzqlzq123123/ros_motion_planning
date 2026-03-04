# TD3 Baseline Model Configuration

## 模型架构: 2层全连接 (2-Fully-Connected)

### Actor 网络
```
Input: state_dim (24)
  -> Linear(24, 800) -> ReLU
  -> Linear(800, 600) -> ReLU
  -> Linear(600, action_dim) -> Tanh
Output: action_dim (2)
```

### Critic 网络 (双Q)
```
分支1 (Q1):
  Input: state_dim (24)
    -> Linear(24, 800) -> ReLU
    -> Linear(800, 600) [state] + Linear(action_dim, 600) [action]
    -> Add + ReLU
    -> Linear(600, 1)
  Output: Q1 value

分支2 (Q2):
  Input: state_dim (24)
    -> Linear(24, 800) -> ReLU
    -> Linear(800, 600) [state] + Linear(action_dim, 600) [action]
    -> Add + ReLU
    -> Linear(600, 1)
  Output: Q2 value
```

## 参数统计

| 网络 | 总参数量 |
|------|---------|
| Actor | ~609,602 |
| Critic | ~2,419,201 (双Q) |

## 实验数据

| Run | 日期 | 说明 |
|-----|------|------|
| run_7 | 2026-02-02 | 基线实验 |
| run_9 | 2026-02-25 | 基线实验 |
| run_10 | 2026-02-27 | 基线实验 |

## 原始模型文件

模型定义: `src/rl_training/third_party/drl_td3/td3_models.py`

## 对照实验说明

此为基线配置 (2层全连接: 800->600)。后续对照实验将增加全连接层数，例如:
- 3层: 800->600->400
- 4层: 800->600->400->200
