# TD3 实验索引

## 目录结构

```
experiments/
├── baseline_2fc/           # 基线: 2层全连接 (800->600)
│   ├── model_config.md     # 模型配置说明
│   ├── td3_models_baseline.py  # 模型代码备份
│   ├── run_7/              # 实验数据
│   ├── run_9/
│   └── run_10/
└── exp_3fc/                # 对照组: 3层全连接 (待创建)
```

## 如何添加新对照实验

1. 修改 `src/rl_training/third_party/drl_td3/td3_models.py` 中的网络结构
2. 创建新的实验目录: `experiments/exp_3fc/`
3. 运行训练
4. 复制数据到新目录并更新配置记录

## 网络层数修改位置

在 `td3_models.py` 中:

**Actor:**
```python
self.layer_1 = nn.Linear(state_dim, 800)
self.layer_2 = nn.Linear(800, 600)
# 添加 layer_3, layer_4...
self.layer_out = nn.Linear(last_hidden, action_dim)
```

**Critic:**
```python
# 同样增加隐藏层
```
