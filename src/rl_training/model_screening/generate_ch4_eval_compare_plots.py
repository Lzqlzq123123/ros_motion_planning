#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

SRC_BASE = Path('/data/lzq/ros_motion_planning/src/rl_training/data')
OUT_BASE = Path('/data/lzq/tjuthesis/figures/paper_data_audit_20260324/fig_4_train_eval_compare')
OUT_BASE.mkdir(parents=True, exist_ok=True)

GROUPS = {
    '无对抗训练': ['run_noadv_nocurr_seed1', 'run_noadv_nocurr_seed2'],
    '有对抗训练（本文）': ['run_1', 'run_13'],
}
COLORS = {
    '无对抗训练': '#4e79a7',
    '有对抗训练（本文）': '#e15759',
}


def set_plot_style():
    font_candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/data/lzq/.local/share/fonts/windows/simsun.ttc',
    ]
    chosen = None
    for p in font_candidates:
        if Path(p).exists():
            try:
                if hasattr(font_manager.fontManager, 'addfont'):
                    font_manager.fontManager.addfont(p)
            except Exception:
                pass
            chosen = font_manager.FontProperties(fname=p).get_name()
            break
    plt.rcParams['font.family'] = chosen or 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = [chosen or 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def load_metric(run, filename, value_transform=lambda x: x):
    path = SRC_BASE / run / filename
    rows = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            step = int(float(row['step']))
            wall_time = float(row.get('wall_time', 0) or 0)
            value = value_transform(float(row['value']))
            rows.append((step, wall_time, value))
    rows.sort(key=lambda x: (x[0], x[1]))
    ded = {}
    for step, wall_time, value in rows:
        ded[step] = value
    steps = np.array(sorted(ded.keys()), dtype=float)
    vals = np.array([ded[s] for s in steps], dtype=float)
    return steps, vals


def aggregate_group(filename, value_transform=lambda x: x):
    group = {}
    for label, runs in GROUPS.items():
        series = []
        all_steps = set()
        for run in runs:
            steps, vals = load_metric(run, filename, value_transform)
            series.append((steps, vals))
            all_steps.update(steps.tolist())
        common = np.array(sorted(all_steps), dtype=float)
        aligned = np.full((len(series), len(common)), np.nan, dtype=float)
        for i, (steps, vals) in enumerate(series):
            mask = (common >= steps.min()) & (common <= steps.max())
            aligned[i, mask] = np.interp(common[mask], steps, vals)
        valid = ~np.all(np.isnan(aligned), axis=0)
        common = common[valid]
        aligned = aligned[:, valid]
        group[label] = {
            'steps': common,
            'mean': np.nanmean(aligned, axis=0),
            'min': np.nanmin(aligned, axis=0),
            'max': np.nanmax(aligned, axis=0),
        }
    return group


def steps_to_train_steps(eval_rounds):
    return eval_rounds * 5000.0


def plot_eval_reward():
    set_plot_style()
    data = aggregate_group('eval_avg_reward.csv')
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=240)
    for label, g in data.items():
        x = steps_to_train_steps(g['steps'])
        ax.plot(x, g['mean'], lw=2.2, color=COLORS[label], label=label)
        ax.fill_between(x, g['min'], g['max'], color=COLORS[label], alpha=0.16)
    ax.set_xlabel('训练步数')
    ax.set_ylabel('评估平均奖励')
    ax.grid(alpha=0.18, linestyle='--')
    ax.legend(frameon=False, fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(OUT_BASE / 'eval_reward_compare.png', bbox_inches='tight')
    try:
        fig.savefig(OUT_BASE / 'eval_reward_compare.pdf', bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)
    return data


def plot_collision_success():
    set_plot_style()
    col = aggregate_group('eval_collision_rate.csv', lambda x: x * 100.0)
    succ = aggregate_group('eval_success_count.csv', lambda x: x)
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 7.2), dpi=240, sharex=True)
    ax = axes[0]
    for label, g in col.items():
        x = steps_to_train_steps(g['steps'])
        ax.plot(x, g['mean'], lw=2.2, color=COLORS[label], label=label)
        ax.fill_between(x, g['min'], g['max'], color=COLORS[label], alpha=0.16)
    ax.set_ylabel('碰撞率 / %')
    ax.grid(alpha=0.18, linestyle='--')
    ax.legend(frameon=False, fontsize=9, loc='best')

    ax = axes[1]
    for label, g in succ.items():
        x = steps_to_train_steps(g['steps'])
        ax.plot(x, g['mean'], lw=2.2, color=COLORS[label], label=label)
        ax.fill_between(x, g['min'], g['max'], color=COLORS[label], alpha=0.16)
    ax.set_xlabel('训练步数')
    ax.set_ylabel('成功次数 / 10')
    ax.grid(alpha=0.18, linestyle='--')
    fig.tight_layout()
    fig.savefig(OUT_BASE / 'eval_collision_success_compare.png', bbox_inches='tight')
    try:
        fig.savefig(OUT_BASE / 'eval_collision_success_compare.pdf', bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)
    return col, succ


def save_summary(reward, col, succ):
    rows = []
    for label in GROUPS:
        rows.append({
            'group': label,
            'reward_last_mean': round(float(reward[label]['mean'][-1]), 4),
            'collision_last_mean_percent': round(float(col[label]['mean'][-1]), 4),
            'success_last_mean_count': round(float(succ[label]['mean'][-1]), 4),
            'num_eval_points': int(len(reward[label]['steps'])),
        })
    with open(OUT_BASE / 'eval_curve_summary.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    reward = plot_eval_reward()
    col, succ = plot_collision_success()
    save_summary(reward, col, succ)
    (OUT_BASE / Path(__file__).name).write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    print('saved to', OUT_BASE)
