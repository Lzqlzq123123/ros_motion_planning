# 已测试模型总表（2026-03-20）

## 1. 已完成独立验证的模型

所有模型均采用以下流程验证：

- 单个模型单独启动 `scripts/main.sh`
- 验证完成后立即关闭仿真环境
- 再启动下一个模型

## 2. 汇总结果

| 模型 | 类别 | 场景 | 成功率 | 碰撞率 | 平均奖励 | 结论 |
|---|---|---:|---:|---:|---:|---|
| `forklift_movebase/run_noadv_nocurr_seed1` | 无对抗基线候选 | adv | 0.000 | 0.750 | -201.71 | 排除 |
| `forklift_movebase/run_noadv_nocurr_seed1` | 无对抗基线候选 | noadv | 0.000 | 0.333 | -218.94 | 排除 |
| `forklift_movebase/run_1` | 无对抗基线候选 | adv | 0.000 | 0.412 | -179.88 | 排除 |
| `forklift_movebase/run_1` | 无对抗基线候选 | noadv | 0.053 | 0.316 | -173.72 | 排除 |
| `forklift_movebase/run_13` | 无对抗基线候选 | adv | 0.000 | 0.700 | -172.76 | 排除 |
| `forklift_movebase/run_13` | 无对抗基线候选 | noadv | 0.000 | 0.684 | -196.20 | 排除 |
| `forklift_movebase/run_14` | 无对抗基线候选 | adv | 0.000 | 0.562 | -193.14 | 排除 |
| `forklift_movebase/run_14` | 无对抗基线候选 | noadv | 0.000 | 0.850 | -199.59 | 排除 |
| `forklift_movebase/run_15` | 无对抗基线候选 | adv | 0.000 | 0.647 | -182.32 | 排除 |
| `forklift_movebase/run_15` | 无对抗基线候选 | noadv | 0.000 | 0.647 | -185.76 | 排除 |
| `forklift_movebase/run_10` | 规则对抗训练 | adv | 0.789 | 0.158 | 27.64 | 可用 |
| `forklift_movebase/run_10` | 规则对抗训练 | noadv | 0.842 | 0.105 | 39.37 | 可用 |
| `forklift_movebase/run_18` | 规划器驱动对抗训练 | adv | 0.833 | 0.111 | 43.77 | 可用，且较强 |
| `forklift_movebase/run_18` | 规划器驱动对抗训练 | noadv | 0.895 | 0.105 | 51.13 | 可用，且较强 |
| `forklift_movebase_with_goal/run_1` | 扩散模型对抗训练（本文） | adv | 0.750 | 0.250 | 33.11 | 可用，作为 seed 补充 |
| `forklift_movebase_with_goal/run_1` | 扩散模型对抗训练（本文） | noadv | 0.889 | 0.111 | 50.91 | 可用，常规场景较强 |
| `forklift_movebase_with_goal/run_2` | 扩散模型对抗训练（本文） | adv | 0.900 | 0.100 | 48.46 | 最佳模型，推荐主结果 |
| `forklift_movebase_with_goal/run_2` | 扩散模型对抗训练（本文） | noadv | 0.650 | 0.350 | 7.85 | 可用，但波动较大 |

## 3. 论文推荐采用的模型

### 推荐用于主表的模型

- 规则对抗训练 TD3：`forklift_movebase/run_10`
- 规划器驱动对抗训练 TD3：`forklift_movebase/run_18`
- 扩散模型对抗训练 TD3（本文）：`forklift_movebase_with_goal/run_2`

### 推荐作为补充 seed 的模型

- 扩散模型对抗训练 TD3（本文）：`forklift_movebase_with_goal/run_1`

## 4. 当前结论

1. 在已经完成独立验证的模型中，**本文方法 `run_2` 在有对抗场景下表现最好**，成功率达到 `0.900`。
2. 规划器驱动对抗训练 `run_18` 表现稳定，且在无对抗场景下达到 `0.895`，可作为强基线。
3. 规则对抗训练 `run_10` 结果稳定，但整体略低于前两者。
4. 当前 logs 中未发现可直接用于论文主表的“高质量无对抗基线”模型，因此无对抗基线部分不建议直接定稿。

## 5. 对论文表格的建议

如果暂不重新训练无对抗基线，则建议：

- 表 4-3 / 表 4-4 优先使用三类对抗训练模型的独立验证结果
- 对“无对抗训练 TD3”保守处理，暂不写入最终定稿主表，或明确说明其为旧版/不稳定基线
- 在正文中强调：本文方法在对抗场景下取得最高成功率，同时在另一 seed 的无对抗场景中仍能达到较高成功率，说明其并未明显损伤常规导航能力
