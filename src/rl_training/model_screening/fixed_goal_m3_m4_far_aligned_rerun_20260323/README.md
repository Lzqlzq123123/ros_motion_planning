# fixed_goal_m3_m4_far_aligned_rerun_20260323

- 固定终点：`(-3, -4)`
- 随机场景，但使用统一 `episode_setups.json` 固定每个 episode 的：
  - Ego 起点
  - 对抗车起点
  - Goal
- 生成 bank 时设置：
  - `ego_spawn_radius = [3.0, 4.0]`
  - `adv_spawn_radius = [3.0, 4.0]`
- 因此不同方法轨迹图可直接横向比较，起点一致。

## 关键文件
- 场景 bank：`bank/episode_setups.json`
- 起点一致性检查：`aligned_start_check.csv`
- 每回合筛图指标：`scene_episode_metrics.csv`
- 候选图：`candidate_overlays/`

## 当前推荐候选
- `candidate_overlays/four_method_ep004_sharedstart.png`
- `candidate_overlays/four_method_ep005_sharedstart.png`

## 本轮整体结果（12 episodes）
- 规则对抗：成功率 50.0%，碰撞率 50.0%
- 规划器驱动对抗：成功率 58.3%，碰撞率 25.0%
- 无条件扩散对抗：成功率 16.7%，碰撞率 58.3%
- 本文方法：成功率 8.3%，碰撞率 91.7%
