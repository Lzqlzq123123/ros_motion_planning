# 数据清理清单（2026-03-20）

## 处理原则

1. 论文当前最终版本直接使用的数据保留在 `raw_csv/`；
2. 原 `eval_csv/` 中仍有参考价值的实验记录，统一快照到 `original_eval_csv_snapshot/`；
3. 明显无关或过时、容易误导当前论文口径的文件予以删除。

## 从 eval_csv 迁移保留到 paper_results_20260320/original_eval_csv_snapshot 的文件

- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/原始无对抗.csv` -> `/data/lzq/ros_motion_planning/src/rl_training/model_screening/paper_results_20260320/original_eval_csv_snapshot/原始无对抗.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/原始有对抗.csv` -> `/data/lzq/ros_motion_planning/src/rl_training/model_screening/paper_results_20260320/original_eval_csv_snapshot/原始有对抗.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/规划有对抗.csv` -> `/data/lzq/ros_motion_planning/src/rl_training/model_screening/paper_results_20260320/original_eval_csv_snapshot/规划有对抗.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/规则无对抗.csv` -> `/data/lzq/ros_motion_planning/src/rl_training/model_screening/paper_results_20260320/original_eval_csv_snapshot/规则无对抗.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/规则有对抗.csv` -> `/data/lzq/ros_motion_planning/src/rl_training/model_screening/paper_results_20260320/original_eval_csv_snapshot/规则有对抗.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/run9_无对抗随机位置.csv` -> `/data/lzq/ros_motion_planning/src/rl_training/model_screening/paper_results_20260320/original_eval_csv_snapshot/run9_无对抗随机位置.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/with_goal_run2_无对抗随机位置.csv` -> `/data/lzq/ros_motion_planning/src/rl_training/model_screening/paper_results_20260320/original_eval_csv_snapshot/with_goal_run2_无对抗随机位置.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/with_goal_run2有对抗随机.csv` -> `/data/lzq/ros_motion_planning/src/rl_training/model_screening/paper_results_20260320/original_eval_csv_snapshot/with_goal_run2有对抗随机.csv`

## 已从 eval_csv 删除的文件

- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/原始无对抗.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/原始有对抗.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/规划有对抗.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/规则无对抗.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/规则有对抗.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/run9_无对抗随机位置.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/with_goal_run2_无对抗随机位置.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/with_goal_run2有对抗随机.csv`
- `/data/lzq/ros_motion_planning/src/rl_training/eval_csv/eval_results.csv`

## 已从 paper_results_20260320 删除的过时文件

- `/data/lzq/ros_motion_planning/src/rl_training/model_screening/paper_results_20260320/selected_model_seed_breakdown.csv`
