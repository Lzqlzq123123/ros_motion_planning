#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import yaml

SRC_BASE = Path('/data/lzq/ros_motion_planning/src/rl_training/data')
LOG_BASE = Path('/data/lzq/ros_motion_planning/src/rl_training/logs/forklift_movebase')
CFG_PATH = Path('/data/lzq/ros_motion_planning/src/rl_training/config/forklift_movebase.yaml')
OUT_BASE = Path('/data/lzq/tjuthesis/figures/paper_data_audit_20260324/table_4_curriculum_stage_metrics')
OUT_BASE.mkdir(parents=True, exist_ok=True)

CURRICULUM_RUNS = ['run_1', 'run_13']
NO_CURR_RUNS = ['run_noadv_nocurr_seed1', 'run_noadv_nocurr_seed2']
EVAL_EPISODES = 10.0

STAGE_BINS = [
    ('阶段 I', '低难度', 1.0, 2.0),
    ('阶段 II', '中低难度', 2.0, 3.0),
    ('阶段 III', '中高难度', 3.0, 4.0),
    ('阶段 IV', '高难度/全难度', 4.0, 5.05),
]


def set_plot_style():
    font_candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/data/lzq/.local/share/fonts/windows/simsun.ttc',
    ]
    chosen_name = None
    chosen_path = None
    for font_path in font_candidates:
        if Path(font_path).exists():
            chosen_path = font_path
            try:
                if hasattr(font_manager.fontManager, 'addfont'):
                    font_manager.fontManager.addfont(font_path)
            except Exception:
                pass
            chosen_name = font_manager.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams['font.family'] = chosen_name or 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = [chosen_name or 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def load_dedup_metric(csv_path: Path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            step = int(float(row['step']))
            wall_time = float(row.get('wall_time', 0) or 0)
            value = float(row['value'])
            rows.append((step, wall_time, value))
    rows.sort(key=lambda x: (x[0], x[1]))
    dedup = {}
    for step, wall_time, value in rows:
        dedup[step] = value
    return dedup


def load_goal_rect_dims():
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    goal_range = cfg['env']['goal_range']
    width = float(goal_range['x_max']) - float(goal_range['x_min'])
    height = float(goal_range['y_max']) - float(goal_range['y_min'])
    return width, height


def classify_stage(span):
    for stage_id, stage_desc, lo, hi in STAGE_BINS:
        if lo <= span < hi or (stage_id == '阶段 IV' and lo <= span <= hi):
            return stage_id, stage_desc, lo, hi
    return '阶段 IV', '高难度/全难度', 4.0, 5.05


def write_csv(path: Path, fieldnames, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    arr = np.asarray(list(values), dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def collect_run_records(run: str, training_mode: str, width0: float, height0: float):
    span_map = load_dedup_metric(SRC_BASE / run / 'train_curriculum_span.csv')
    succ_map = load_dedup_metric(SRC_BASE / run / 'eval_success_count.csv')
    coll_map = load_dedup_metric(SRC_BASE / run / 'eval_collision_rate.csv')
    reward_map = load_dedup_metric(SRC_BASE / run / 'eval_avg_reward.csv')

    common_steps = sorted(set(span_map) & set(succ_map) & set(coll_map) & set(reward_map))
    records = []
    for step in common_steps:
        span = float(span_map[step])
        stage_id, stage_desc, lo, hi = classify_stage(span)
        area = (width0 + 2.0 * span) * (height0 + 2.0 * span)
        diag = math.sqrt((width0 + 2.0 * span) ** 2 + (height0 + 2.0 * span) ** 2)
        records.append({
            'training_mode': training_mode,
            'run': run,
            'eval_step': step,
            'curriculum_span': span,
            'stage_id': stage_id,
            'stage_desc': stage_desc,
            'stage_span_lo': lo,
            'stage_span_hi': hi,
            'task_space_area': area,
            'task_space_area_ratio_vs_base': area / (width0 * height0),
            'task_space_diag': diag,
            'task_space_diag_ratio_vs_base': diag / math.sqrt(width0 ** 2 + height0 ** 2),
            'eval_success_count': float(succ_map[step]),
            'eval_success_rate_percent': float(succ_map[step]) / EVAL_EPISODES * 100.0,
            'eval_collision_rate_percent': float(coll_map[step]) * 100.0,
            'eval_avg_reward': float(reward_map[step]),
        })
    return records


def summarize_stage(records, group_cols):
    groups = {}
    for row in records:
        key = tuple(row[col] for col in group_cols)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, rows in groups.items():
        out = {col: key[i] for i, col in enumerate(group_cols)}
        out['num_eval_points'] = len(rows)
        for metric in [
            'curriculum_span',
            'task_space_area',
            'task_space_area_ratio_vs_base',
            'task_space_diag',
            'task_space_diag_ratio_vs_base',
            'eval_success_rate_percent',
            'eval_collision_rate_percent',
            'eval_avg_reward',
        ]:
            m, s = mean_std([r[metric] for r in rows])
            out[f'{metric}_mean'] = m
            out[f'{metric}_std'] = s
        summary.append(out)

    def sort_key(r):
        order = {name: idx for idx, (name, *_rest) in enumerate(STAGE_BINS)}
        return (r.get('training_mode', ''), r.get('run', ''), order.get(r.get('stage_id', '阶段 IV'), 99))

    summary.sort(key=sort_key)
    return summary


def plot_stage_summary(curr_summary, nocurr_summary):
    set_plot_style()
    stage_order = [x[0] for x in STAGE_BINS]
    x = np.arange(len(stage_order))

    span = [next(r for r in curr_summary if r['stage_id'] == s)['curriculum_span_mean'] for s in stage_order]
    area_ratio = [next(r for r in curr_summary if r['stage_id'] == s)['task_space_area_ratio_vs_base_mean'] for s in stage_order]
    succ = [next(r for r in curr_summary if r['stage_id'] == s)['eval_success_rate_percent_mean'] for s in stage_order]
    coll = [next(r for r in curr_summary if r['stage_id'] == s)['eval_collision_rate_percent_mean'] for s in stage_order]

    nocurr_succ = nocurr_summary['eval_success_rate_percent_mean']
    nocurr_coll = nocurr_summary['eval_collision_rate_percent_mean']

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=240)
    ax.plot(x, span, marker='o', color='#4e79a7', lw=2.2, label='课程跨度')
    ax2 = ax.twinx()
    ax2.plot(x, area_ratio, marker='s', color='#e15759', lw=2.2, label='任务空间面积倍率')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_order)
    ax.set_ylabel('课程跨度')
    ax2.set_ylabel('面积倍率')
    ax.grid(alpha=0.18, linestyle='--')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_BASE / 'curriculum_difficulty_progress.png', bbox_inches='tight')
    try:
        fig.savefig(OUT_BASE / 'curriculum_difficulty_progress.pdf', bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=240)
    ax.plot(x, succ, marker='o', color='#59a14f', lw=2.2, label='课程学习成功率')
    ax.plot(x, coll, marker='s', color='#f28e2b', lw=2.2, label='课程学习碰撞率')
    ax.axhline(nocurr_succ, color='#59a14f', lw=1.4, ls='--', alpha=0.7, label='无课程成功率')
    ax.axhline(nocurr_coll, color='#f28e2b', lw=1.4, ls='--', alpha=0.7, label='无课程碰撞率')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_order)
    ax.set_ylabel('百分比 / %')
    ax.grid(alpha=0.18, linestyle='--')
    ax.legend(frameon=False, loc='best', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_BASE / 'curriculum_stage_performance_compare.png', bbox_inches='tight')
    try:
        fig.savefig(OUT_BASE / 'curriculum_stage_performance_compare.pdf', bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=240)

    ax = axes[0]
    ax.plot(x, span, marker='o', color='#4e79a7', lw=2.2, label='课程跨度')
    ax2 = ax.twinx()
    ax2.plot(x, area_ratio, marker='s', color='#e15759', lw=2.2, label='任务空间面积倍率')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_order)
    ax.set_ylabel('课程跨度')
    ax2.set_ylabel('面积倍率')
    ax.grid(alpha=0.18, linestyle='--')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='upper left', fontsize=9)

    ax = axes[1]
    ax.plot(x, succ, marker='o', color='#59a14f', lw=2.2, label='课程学习成功率')
    ax.plot(x, coll, marker='s', color='#f28e2b', lw=2.2, label='课程学习碰撞率')
    ax.axhline(nocurr_succ, color='#59a14f', lw=1.4, ls='--', alpha=0.7, label='无课程成功率')
    ax.axhline(nocurr_coll, color='#f28e2b', lw=1.4, ls='--', alpha=0.7, label='无课程碰撞率')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_order)
    ax.set_ylabel('百分比 / %')
    ax.grid(alpha=0.18, linestyle='--')
    ax.legend(frameon=False, loc='best', fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_BASE / 'curriculum_stage_metrics_overview.png', bbox_inches='tight')
    try:
        fig.savefig(OUT_BASE / 'curriculum_stage_metrics_overview.pdf', bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)


def main():
    width0, height0 = load_goal_rect_dims()

    all_records = []
    for run in CURRICULUM_RUNS:
        all_records.extend(collect_run_records(run, '课程学习训练', width0, height0))
    for run in NO_CURR_RUNS:
        all_records.extend(collect_run_records(run, '无课程学习训练', width0, height0))

    per_run_summary = summarize_stage(all_records, ['training_mode', 'run', 'stage_id', 'stage_desc'])
    group_summary = summarize_stage(all_records, ['training_mode', 'stage_id', 'stage_desc'])

    curriculum_group = [r for r in group_summary if r['training_mode'] == '课程学习训练']
    nocurr_group = [r for r in group_summary if r['training_mode'] == '无课程学习训练']
    nocurr_full = max(nocurr_group, key=lambda r: r['curriculum_span_mean'])

    merged_fields = list(all_records[0].keys())
    write_csv(OUT_BASE / 'curriculum_eval_merged_all_runs.csv', merged_fields, all_records)

    per_run_fields = list(per_run_summary[0].keys())
    write_csv(OUT_BASE / 'curriculum_stage_summary_per_run.csv', per_run_fields, per_run_summary)

    group_fields = list(group_summary[0].keys())
    write_csv(OUT_BASE / 'curriculum_stage_summary_group.csv', group_fields, group_summary)

    compare_rows = []
    curr_stage_iv = next(r for r in curriculum_group if r['stage_id'] == '阶段 IV')
    compare_rows.append({
        'comparison': '课程阶段IV vs 无课程全难度',
        'curriculum_success_rate_mean': curr_stage_iv['eval_success_rate_percent_mean'],
        'nocurr_success_rate_mean': nocurr_full['eval_success_rate_percent_mean'],
        'success_rate_gain': curr_stage_iv['eval_success_rate_percent_mean'] - nocurr_full['eval_success_rate_percent_mean'],
        'curriculum_collision_rate_mean': curr_stage_iv['eval_collision_rate_percent_mean'],
        'nocurr_collision_rate_mean': nocurr_full['eval_collision_rate_percent_mean'],
        'collision_rate_reduction': nocurr_full['eval_collision_rate_percent_mean'] - curr_stage_iv['eval_collision_rate_percent_mean'],
        'curriculum_reward_mean': curr_stage_iv['eval_avg_reward_mean'],
        'nocurr_reward_mean': nocurr_full['eval_avg_reward_mean'],
        'reward_gain': curr_stage_iv['eval_avg_reward_mean'] - nocurr_full['eval_avg_reward_mean'],
    })
    write_csv(OUT_BASE / 'curriculum_vs_nocurr_full_difficulty_compare.csv', list(compare_rows[0].keys()), compare_rows)

    plot_stage_summary(curriculum_group, nocurr_full)

    with open(OUT_BASE / 'README_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(
            '课程学习阶段指标说明\n'
            '1. 课程跨度：直接来自 train_curriculum_span.csv，表示当前目标采样范围扩展尺度。\n'
            '2. 任务空间面积：基于训练配置中的基础 goal_range，按 (W0+2S)(H0+2S) 计算。\n'
            '3. 任务空间对角线：基于训练配置中的基础 goal_range，按 sqrt((W0+2S)^2+(H0+2S)^2) 计算。\n'
            '4. 成功率：eval_success_count / 10。\n'
            '5. 碰撞率：eval_collision_rate * 100%。\n'
            '6. 本目录用于课程学习难度递增与阶段效果分析的第一版统计。\n'
        )

    (OUT_BASE / Path(__file__).name).write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    print('saved to', OUT_BASE)


if __name__ == '__main__':
    main()
