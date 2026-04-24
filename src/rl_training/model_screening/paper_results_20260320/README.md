# 论文主表结果整理包（当前最终版）

生成时间：2026-03-21

## 本次整理目的

将论文**当前最终采用**的主表结果（含表 3-1、表 4-3、表 4-4）与其对应的源 CSV、训练模型目录、重复验证结果统一整理，便于后续核查。

> 如需继续接手 diffusion / `main.sh` / 表 3-1 随机评测流程，请先阅读：
> `../README_DIFFUSION_HANDOFF.md`

## 当前已整理表格

- 表 3-1：不同对抗方法下 Ego 车威胁性指标对比
- 表 4-3：不同训练方式在两类验证场景下的任务成功率与波动幅度对比
- 表 4-4：训练完成后在有对抗验证场景下各指标均值对比

## 表 3-1 当前最终采用

- 规则对抗（Rule-based）：37.9 ± 1.8 / 1.56E-04 / 3.21 / 3.18
- 规划器驱动对抗（Planner-based）：31.9 ± 14.8 / 7.39E-05 / 3.84 / 2.56
- 扩散模型对抗（Ours）：55.2 ± 21.5 / 9.08E-05 / 2.31 / 2.87

说明：表 3-1 第一列采用 3 次独立随机验证的均值±标准差，相关统计见 `table_3_1_collision_repeat_stats.csv`；其余指标仍采用代表性随机验证结果，相关原始 CSV 保存在 `table_3_1_random_selected/` 与 `table_3_1_movebase_repeats/`。

## 目录说明

- `raw_csv/`：当前最终表格实际采用的源 CSV 拷贝
- `original_eval_csv_snapshot/`：从 `eval_csv/` 迁移保留的历史实验 CSV 快照
- `table_3_1_summary.csv`：表 3-1 当前最终汇总
- `table_3_1_traceability.csv`：表 3-1 结果追溯表
- `table_3_1_supporting_csv/`：表 3-1 早期支撑 CSV
- `table_3_1_movebase_repeats/`：表 3-1 当前最终采用的 movebase 统一评测原始 CSV 与 summary
- `table_4_3_summary.csv`：表 4-3 当前最终汇总（宽表）
- `table_4_3_traceability.csv`：表 4-3 逐项追溯表，含源文件与模型路径
- `table_4_4_summary.csv`：表 4-4 当前最终汇总
- `paper_results_manifest.json`：机器可读的结果追溯清单

## 表 4-3 当前最终采用

- 历史参考对照 TD3：72.8 -> 50.0，波动 22.8
- 规则对抗训练 TD3：84.2 -> 78.9，波动 5.3
- 规划器驱动对抗训练 TD3：89.5 -> 83.3，波动 6.2
- 扩散模型对抗训练 TD3（本文，run_1）：90.8 -> 80.5，波动 10.3

## 表 4-4 当前最终采用

- 评估平均奖励：37.1 vs -37.8
- 碰撞率：19.5% vs 47.1%
- 任务成功次数（/10）：8.1 vs 5.0

## 特别说明

1. “历史参考对照 TD3”来自旧版 `eval_csv/原始无对抗.csv` 与 `eval_csv/原始有对抗.csv`，当前仅作为历史参考对照项；
2. 本文方法不再采用“2 seeds 平均”作为主表结果，当前统一使用 `forklift_movebase_with_goal/run_1`；
3. 其中本文方法有对抗结果采用 `ours_run1_adv_native_repeat_aggregate.csv`，是基于模型自身训练环境参数进行的 3 次独立重测均值；
4. 本文方法无对抗结果采用 `ours_run1_noadv_repeat_aggregate.csv`，对应 3 次独立验证均值。
