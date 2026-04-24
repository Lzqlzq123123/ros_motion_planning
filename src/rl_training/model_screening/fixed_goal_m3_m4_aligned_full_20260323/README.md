# fixed_goal_m3_m4_aligned_full_20260323

- 固定终点: (-3, -4)
- 先生成带 `ego+adv` 全部初始位姿的场景 bank: `bank/episode_setups.json`
- 再用同一 bank 重跑 rule / planner / unconditional / ours / noadv
- 因此这一次不同方法的起点是一致的，可直接横向比较。

## 推荐先看这些候选图
- candidate_overlays/four_method_ep003.png
- candidate_overlays/four_method_ep005.png
- candidate_overlays/four_method_ep008.png
- candidate_overlays/four_method_ep011.png

## 对齐校验
- `aligned_start_check.csv`：可检查每个 episode 各方法的 ego/adv 初始位置是否一致
