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
CFG_PATH = Path('/data/lzq/ros_motion_planning/src/rl_training/config/forklift_movebase.yaml')
OUT_BASE = Path('/data/lzq/tjuthesis/figures/paper_data_audit_20260324/table_4_curriculum_stability_analysis')
OUT_BASE.mkdir(parents=True, exist_ok=True)

CURRICULUM_RUNS = ['run_1', 'run_13']
NO_CURR_RUNS = ['run_noadv_nocurr_seed1', 'run_noadv_nocurr_seed2']
EVAL_EPISODES = 10.0
SMOOTH_WEIGHT = 0.9
STAGE_BINS = [
    ('阶段 I', '低难度', 1.0, 2.0),
    ('阶段 II', '中低难度', 2.0, 3.0),
    ('阶段 III', '中高难度', 3.0, 4.0),
    ('阶段 IV', '高难度/全难度', 4.0, 5.05),
]
STAGE_ORDER = {name: idx for idx, (name, *_rest) in enumerate(STAGE_BINS)}
TRANSITIONS = [('阶段 I', '阶段 II'), ('阶段 II', '阶段 III'), ('阶段 III', '阶段 IV')]
RECOVERY_HOLD_EVAL = 1
RECOVERY_RATIO = 0.90
Q_RECOVERY_HOLD = 50


def set_plot_style():
    font_candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/data/lzq/.local/share/fonts/windows/simsun.ttc',
    ]
    chosen_name = None
    for font_path in font_candidates:
        if Path(font_path).exists():
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


def write_csv(path: Path, fieldnames, rows):
    fields = list(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ewm(values, weight=SMOOTH_WEIGHT):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1.0 - weight) * values[i] + weight * out[i - 1]
    return out


def load_series(csv_path: Path, smooth=False):
    rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows.append({
                'step': int(float(row['step'])),
                'wall_time': float(row.get('wall_time', 0) or 0),
                'value': float(row['value']),
            })
    rows.sort(key=lambda r: (r['step'], r['wall_time']))
    dedup = {}
    for row in rows:
        dedup[row['step']] = row
    series = [dedup[k] for k in sorted(dedup)]
    if smooth and series:
        smoothed = ewm([r['value'] for r in series])
        for row, s in zip(series, smoothed):
            row['smoothed_value'] = float(s)
    return series


def classify_stage(span):
    for stage_id, stage_desc, lo, hi in STAGE_BINS:
        if lo <= span < hi or (stage_id == '阶段 IV' and lo <= span <= hi):
            return stage_id, stage_desc, lo, hi
    return '阶段 IV', '高难度/全难度', 4.0, 5.05


def load_base_dims():
    with open(CFG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    goal_range = cfg['env']['goal_range']
    width0 = float(goal_range['x_max']) - float(goal_range['x_min'])
    height0 = float(goal_range['y_max']) - float(goal_range['y_min'])
    return width0, height0


def build_eval_records(run, training_mode, width0, height0):
    span_series = load_series(SRC_BASE / run / 'train_curriculum_span.csv')
    succ_series = load_series(SRC_BASE / run / 'eval_success_count.csv')
    coll_series = load_series(SRC_BASE / run / 'eval_collision_rate.csv')
    reward_series = load_series(SRC_BASE / run / 'eval_avg_reward.csv')
    common_steps = sorted(set(r['step'] for r in span_series) & set(r['step'] for r in succ_series) & set(r['step'] for r in coll_series) & set(r['step'] for r in reward_series))
    span_map = {r['step']: r for r in span_series}
    succ_map = {r['step']: r for r in succ_series}
    coll_map = {r['step']: r for r in coll_series}
    reward_map = {r['step']: r for r in reward_series}
    out = []
    for step in common_steps:
        span = float(span_map[step]['value'])
        stage_id, stage_desc, lo, hi = classify_stage(span)
        area = (width0 + 2.0 * span) * (height0 + 2.0 * span)
        diag = math.sqrt((width0 + 2.0 * span) ** 2 + (height0 + 2.0 * span) ** 2)
        out.append({
            'training_mode': training_mode,
            'run': run,
            'eval_step': step,
            'wall_time': float(span_map[step]['wall_time']),
            'curriculum_span': span,
            'stage_id': stage_id,
            'stage_desc': stage_desc,
            'stage_span_lo': lo,
            'stage_span_hi': hi,
            'task_space_area': area,
            'task_space_area_ratio_vs_base': area / (width0 * height0),
            'task_space_diag': diag,
            'eval_success_rate_percent': float(succ_map[step]['value']) / EVAL_EPISODES * 100.0,
            'eval_collision_rate_percent': float(coll_map[step]['value']) * 100.0,
            'eval_avg_reward': float(reward_map[step]['value']),
        })
    return out


def build_stage_segments(eval_records):
    records = sorted(eval_records, key=lambda r: r['eval_step'])
    segments = []
    if not records:
        return segments
    start_idx = 0
    for i in range(1, len(records) + 1):
        if i == len(records) or records[i]['stage_id'] != records[start_idx]['stage_id']:
            seg_rows = records[start_idx:i]
            segments.append({
                'stage_id': seg_rows[0]['stage_id'],
                'stage_desc': seg_rows[0]['stage_desc'],
                'eval_start_step': seg_rows[0]['eval_step'],
                'eval_end_step': seg_rows[-1]['eval_step'],
                'start_time': -np.inf if start_idx == 0 else seg_rows[0]['wall_time'],
                'enter_time': seg_rows[0]['wall_time'],
                'end_time': np.inf if i == len(records) else records[i]['wall_time'],
                'rows': seg_rows,
            })
            start_idx = i
    return segments


def safe_mean(vals):
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def safe_std(vals):
    arr = np.asarray(list(vals), dtype=float)
    if arr.size == 0:
        return np.nan
    return float(np.std(arr))


def first_recovery_index(series, target, hold, mode='ge'):
    if len(series) == 0 or np.isnan(target):
        return None
    for i in range(0, len(series) - hold + 1):
        seg = series[i:i + hold]
        if mode == 'ge':
            ok = np.all(seg >= target)
        else:
            ok = np.all(seg <= target)
        if ok:
            return i
    return None


def summarize_stages_and_transitions(run, training_mode, eval_records):
    segments = build_stage_segments(eval_records)
    maxq_series = load_series(SRC_BASE / run / 'Max._Q.csv', smooth=True)
    avq_series = load_series(SRC_BASE / run / 'Av._Q.csv', smooth=True)

    stage_rows = []
    transition_rows = []
    for idx, seg in enumerate(segments):
        rows = seg['rows']
        q_rows = [r for r in maxq_series if seg['start_time'] <= r['wall_time'] < seg['end_time']]
        av_rows = [r for r in avq_series if seg['start_time'] <= r['wall_time'] < seg['end_time']]

        stage_row = {
            'training_mode': training_mode,
            'run': run,
            'stage_id': seg['stage_id'],
            'stage_desc': seg['stage_desc'],
            'num_eval_points': len(rows),
            'curriculum_span_mean': safe_mean(r['curriculum_span'] for r in rows),
            'task_space_area_ratio_mean': safe_mean(r['task_space_area_ratio_vs_base'] for r in rows),
            'task_space_diag_mean': safe_mean(r['task_space_diag'] for r in rows),
            'eval_success_rate_mean': safe_mean(r['eval_success_rate_percent'] for r in rows),
            'eval_success_rate_std': safe_std(r['eval_success_rate_percent'] for r in rows),
            'eval_collision_rate_mean': safe_mean(r['eval_collision_rate_percent'] for r in rows),
            'eval_collision_rate_std': safe_std(r['eval_collision_rate_percent'] for r in rows),
            'eval_avg_reward_mean': safe_mean(r['eval_avg_reward'] for r in rows),
            'eval_avg_reward_std': safe_std(r['eval_avg_reward'] for r in rows),
            'max_q_smoothed_mean': safe_mean(r['smoothed_value'] for r in q_rows),
            'max_q_smoothed_std': safe_std(r['smoothed_value'] for r in q_rows),
            'av_q_smoothed_mean': safe_mean(r['smoothed_value'] for r in av_rows),
            'av_q_smoothed_std': safe_std(r['smoothed_value'] for r in av_rows),
            'max_q_points_in_stage': len(q_rows),
            'av_q_points_in_stage': len(av_rows),
        }
        stage_rows.append(stage_row)

        if idx == 0:
            continue
        prev_seg = segments[idx - 1]
        prev_stage = prev_seg['stage_id']
        curr_stage = seg['stage_id']
        success_series = np.asarray([r['eval_success_rate_percent'] for r in rows], dtype=float)
        collision_series = np.asarray([r['eval_collision_rate_percent'] for r in rows], dtype=float)
        tail_len = max(2, min(5, len(rows)))
        success_target = safe_mean(success_series[-tail_len:]) * RECOVERY_RATIO
        collision_target = safe_mean(collision_series[-tail_len:]) / RECOVERY_RATIO
        success_recovery = first_recovery_index(success_series, success_target, RECOVERY_HOLD_EVAL, mode='ge')
        collision_recovery = first_recovery_index(collision_series, collision_target, RECOVERY_HOLD_EVAL, mode='le')

        curr_q_rows = [r for r in maxq_series if seg['start_time'] <= r['wall_time'] < seg['end_time']]
        curr_q_vals = np.asarray([r['smoothed_value'] for r in curr_q_rows], dtype=float)
        q_tail_len = max(20, min(100, len(curr_q_vals))) if len(curr_q_vals) else 0
        q_target = safe_mean(curr_q_vals[-q_tail_len:]) * RECOVERY_RATIO if q_tail_len else np.nan
        q_recovery = first_recovery_index(curr_q_vals, q_target, min(Q_RECOVERY_HOLD, len(curr_q_vals)) if len(curr_q_vals) else 1, mode='ge')
        if q_recovery is not None and curr_q_rows:
            q_recovery_episodes = int(curr_q_rows[q_recovery]['step'] - curr_q_rows[0]['step'])
        else:
            q_recovery_episodes = None

        transition_rows.append({
            'training_mode': training_mode,
            'run': run,
            'transition': f'{prev_stage}->{curr_stage}',
            'from_stage': prev_stage,
            'to_stage': curr_stage,
            'enter_eval_step': seg['eval_start_step'],
            'success_target_90pct': success_target,
            'success_recovery_eval_rounds': success_recovery,
            'collision_target_relaxed': collision_target,
            'collision_recovery_eval_rounds': collision_recovery,
            'q_target_90pct': q_target,
            'max_q_recovery_episodes': q_recovery_episodes,
            'target_tail_eval_points': tail_len,
            'target_tail_q_points': q_tail_len,
        })
    return stage_rows, transition_rows, segments


def aggregate_rows(rows, key_cols):
    groups = {}
    for row in rows:
        key = tuple(row[c] for c in key_cols)
        groups.setdefault(key, []).append(row)
    out = []
    for key, grows in groups.items():
        record = {c: key[i] for i, c in enumerate(key_cols)}
        numeric_cols = [c for c in grows[0].keys() if c not in key_cols and isinstance(grows[0][c], (int, float, np.floating))]
        for c in numeric_cols:
            vals = [float(r[c]) for r in grows if r[c] is not None and not np.isnan(r[c])]
            if vals:
                record[f'{c}_mean'] = float(np.mean(vals))
                record[f'{c}_std'] = float(np.std(vals))
        record['num_runs'] = len(grows)
        out.append(record)
    out.sort(key=lambda r: tuple(r.get(c, '') if c != 'stage_id' else STAGE_ORDER.get(r.get(c,''),99) for c in key_cols))
    return out


def build_transition_curve_rows(segments_by_run):
    out = []
    max_window = 8
    for run, segments in segments_by_run.items():
        for idx in range(1, len(segments)):
            prev_stage = segments[idx - 1]['stage_id']
            curr_stage = segments[idx]['stage_id']
            rows = segments[idx]['rows']
            for j, r in enumerate(rows[:max_window]):
                out.append({
                    'run': run,
                    'transition': f'{prev_stage}->{curr_stage}',
                    'to_stage': curr_stage,
                    'relative_eval_round': j,
                    'success_rate_percent': r['eval_success_rate_percent'],
                    'collision_rate_percent': r['eval_collision_rate_percent'],
                    'avg_reward': r['eval_avg_reward'],
                })
    return out


def plot_transition_recovery(curve_rows):
    set_plot_style()
    transitions = [f'{a}->{b}' for a, b in TRANSITIONS]
    colors = {'阶段 I->阶段 II': '#4e79a7', '阶段 II->阶段 III': '#f28e2b', '阶段 III->阶段 IV': '#e15759'}
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=240)
    for tr in transitions:
        rows = [r for r in curve_rows if r['transition'] == tr]
        if not rows:
            continue
        rels = sorted(set(r['relative_eval_round'] for r in rows))
        ys = []
        for rel in rels:
            vals = [r['success_rate_percent'] for r in rows if r['relative_eval_round'] == rel]
            ys.append(np.mean(vals))
        ax.plot(rels, ys, marker='o', lw=2.2, label=tr.replace('->', ' → '), color=colors.get(tr, None))
    ax.set_xlabel('进入新阶段后的评估轮次')
    ax.set_ylabel('成功率 / %')
    ax.grid(alpha=0.18, linestyle='--')
    ax.legend(frameon=False, fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(OUT_BASE / 'curriculum_transition_recovery.png', bbox_inches='tight')
    try:
        fig.savefig(OUT_BASE / 'curriculum_transition_recovery.pdf', bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)


def plot_stage_stability(stage_group_rows, no_curr_row):
    set_plot_style()
    labels = [r['stage_id'] for r in stage_group_rows] + ['无课程全难度']
    success_std = [r['eval_success_rate_std_mean'] for r in stage_group_rows] + [no_curr_row['eval_success_rate_std_mean']]
    collision_std = [r['eval_collision_rate_std_mean'] for r in stage_group_rows] + [no_curr_row['eval_collision_rate_std_mean']]
    maxq_std = [r['max_q_smoothed_std_mean'] for r in stage_group_rows] + [no_curr_row['max_q_smoothed_std_mean']]
    avq_std = [r['av_q_smoothed_std_mean'] for r in stage_group_rows] + [no_curr_row['av_q_smoothed_std_mean']]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=240)
    width = 0.34
    ax.bar(x - width / 2, success_std, width=width, color='#59a14f', label='成功率波动标准差')
    ax.bar(x + width / 2, collision_std, width=width, color='#f28e2b', label='碰撞率波动标准差')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('评估指标波动标准差 / %')
    ax.grid(alpha=0.18, linestyle='--', axis='y')
    ax.legend(frameon=False, fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(OUT_BASE / 'curriculum_eval_stability.png', bbox_inches='tight')
    try:
        fig.savefig(OUT_BASE / 'curriculum_eval_stability.pdf', bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=240)
    width = 0.34
    ax.bar(x - width / 2, maxq_std, width=width, color='#4e79a7', label='Max Q 波动标准差')
    ax.bar(x + width / 2, avq_std, width=width, color='#b07aa1', label='Av Q 波动标准差')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Q 值波动标准差')
    ax.grid(alpha=0.18, linestyle='--', axis='y')
    ax.legend(frameon=False, fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(OUT_BASE / 'curriculum_q_stability.png', bbox_inches='tight')
    try:
        fig.savefig(OUT_BASE / 'curriculum_q_stability.pdf', bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=240)
    width = 0.34
    axes[0].bar(x - width / 2, success_std, width=width, color='#59a14f', label='成功率波动标准差')
    axes[0].bar(x + width / 2, collision_std, width=width, color='#f28e2b', label='碰撞率波动标准差')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('评估指标波动标准差 / %')
    axes[0].grid(alpha=0.18, linestyle='--', axis='y')
    axes[0].legend(frameon=False, fontsize=9, loc='best')
    axes[1].bar(x - width / 2, maxq_std, width=width, color='#4e79a7', label='Max Q 波动标准差')
    axes[1].bar(x + width / 2, avq_std, width=width, color='#b07aa1', label='Av Q 波动标准差')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Q 值波动标准差')
    axes[1].grid(alpha=0.18, linestyle='--', axis='y')
    axes[1].legend(frameon=False, fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(OUT_BASE / 'curriculum_stage_stability.png', bbox_inches='tight')
    try:
        fig.savefig(OUT_BASE / 'curriculum_stage_stability.pdf', bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)


def main():
    width0, height0 = load_base_dims()
    all_eval_rows = []
    all_stage_rows = []
    all_transition_rows = []
    segments_by_run = {}

    for run in CURRICULUM_RUNS:
        eval_rows = build_eval_records(run, '课程学习训练', width0, height0)
        stage_rows, transition_rows, segments = summarize_stages_and_transitions(run, '课程学习训练', eval_rows)
        all_eval_rows.extend(eval_rows)
        all_stage_rows.extend(stage_rows)
        all_transition_rows.extend(transition_rows)
        segments_by_run[run] = segments

    for run in NO_CURR_RUNS:
        eval_rows = build_eval_records(run, '无课程学习训练', width0, height0)
        stage_rows, transition_rows, segments = summarize_stages_and_transitions(run, '无课程学习训练', eval_rows)
        all_eval_rows.extend(eval_rows)
        all_stage_rows.extend(stage_rows)
        all_transition_rows.extend(transition_rows)
        segments_by_run[run] = segments

    transition_curve_rows = build_transition_curve_rows(segments_by_run)
    stage_group_rows = aggregate_rows([r for r in all_stage_rows if r['training_mode'] == '课程学习训练'], ['training_mode', 'stage_id', 'stage_desc'])
    no_curr_group_rows = aggregate_rows([r for r in all_stage_rows if r['training_mode'] == '无课程学习训练'], ['training_mode', 'stage_id', 'stage_desc'])
    no_curr_full = max(no_curr_group_rows, key=lambda r: STAGE_ORDER.get(r['stage_id'], 99))
    transition_group_rows = aggregate_rows(all_transition_rows, ['training_mode', 'transition', 'from_stage', 'to_stage'])

    summary_table = []
    for r in stage_group_rows:
        trans = next((t for t in transition_group_rows if t['to_stage'] == r['stage_id']), None)
        summary_table.append({
            'training_mode': r['training_mode'],
            'stage_id': r['stage_id'],
            'stage_desc': r['stage_desc'],
            'curriculum_span_mean': r.get('curriculum_span_mean_mean'),
            'task_space_area_ratio_mean': r.get('task_space_area_ratio_mean_mean'),
            'eval_success_rate_mean': r.get('eval_success_rate_mean_mean'),
            'eval_collision_rate_mean': r.get('eval_collision_rate_mean_mean'),
            'eval_success_rate_std': r.get('eval_success_rate_std_mean'),
            'max_q_smoothed_std': r.get('max_q_smoothed_std_mean'),
            'success_recovery_eval_rounds': trans.get('success_recovery_eval_rounds_mean') if trans else None,
            'collision_recovery_eval_rounds': trans.get('collision_recovery_eval_rounds_mean') if trans else None,
            'max_q_recovery_episodes': trans.get('max_q_recovery_episodes_mean') if trans else None,
        })
    summary_table.append({
        'training_mode': no_curr_full['training_mode'],
        'stage_id': no_curr_full['stage_id'],
        'stage_desc': no_curr_full['stage_desc'],
        'curriculum_span_mean': no_curr_full.get('curriculum_span_mean_mean'),
        'task_space_area_ratio_mean': no_curr_full.get('task_space_area_ratio_mean_mean'),
        'eval_success_rate_mean': no_curr_full.get('eval_success_rate_mean_mean'),
        'eval_collision_rate_mean': no_curr_full.get('eval_collision_rate_mean_mean'),
        'eval_success_rate_std': no_curr_full.get('eval_success_rate_std_mean'),
        'max_q_smoothed_std': no_curr_full.get('max_q_smoothed_std_mean'),
        'success_recovery_eval_rounds': None,
        'collision_recovery_eval_rounds': None,
        'max_q_recovery_episodes': None,
    })

    write_csv(OUT_BASE / 'curriculum_eval_records_all_runs.csv', list(all_eval_rows[0].keys()), all_eval_rows)
    write_csv(OUT_BASE / 'curriculum_stage_stability_per_run.csv', list(all_stage_rows[0].keys()), all_stage_rows)
    write_csv(OUT_BASE / 'curriculum_stage_stability_group.csv', list(stage_group_rows[0].keys()), stage_group_rows + no_curr_group_rows)
    write_csv(OUT_BASE / 'curriculum_transition_recovery_per_run.csv', list(all_transition_rows[0].keys()), all_transition_rows)
    write_csv(OUT_BASE / 'curriculum_transition_recovery_group.csv', list(transition_group_rows[0].keys()), transition_group_rows)
    write_csv(OUT_BASE / 'curriculum_transition_curve_all_runs.csv', list(transition_curve_rows[0].keys()), transition_curve_rows)
    write_csv(OUT_BASE / 'curriculum_stage_summary_for_paper.csv', list(summary_table[0].keys()), summary_table)

    plot_transition_recovery(transition_curve_rows)
    plot_stage_stability(stage_group_rows, no_curr_full)

    with open(OUT_BASE / 'README_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(
            '课程学习阶段切换恢复速度与阶段内稳定性说明\n'
            '1. 阶段划分依据 curriculum span：I=[1,2), II=[2,3), III=[3,4), IV=[4,5].\n'
            '2. 阶段内波动：分别统计该阶段内评估成功率标准差、评估碰撞率标准差，以及按 wall_time 分段后的 smoothed Max Q / Av Q 标准差。\n'
            '3. 成功率恢复轮次：进入新阶段后，首次达到该阶段后段均值 90% 且连续 2 次评估满足条件的最早评估轮次。\n'
            '4. 碰撞率恢复轮次：进入新阶段后，首次下降到该阶段后段均值/0.9 以内且连续 2 次评估满足条件的最早评估轮次。\n'
            '5. Max Q 恢复回合数：进入新阶段后，按 wall_time 划分该阶段内的 smoothed Max Q 序列，首次达到该阶段后段均值 90% 的最早位置与阶段首点之差。\n'
        )
    (OUT_BASE / Path(__file__).name).write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    print('saved to', OUT_BASE)


if __name__ == '__main__':
    main()
