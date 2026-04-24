#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D


def setup_fonts():
    candidate_fonts = [
        'Noto Sans CJK SC', 'Noto Sans SC', 'Source Han Sans SC',
        'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei',
        'PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS',
    ]
    for font_path in [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    ]:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                pass
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name]
            break
    plt.rcParams['axes.unicode_minus'] = False


def load_best_candidate(debug_step_dir):
    meta = json.load(open(os.path.join(debug_step_dir, 'selection_meta.json'), 'r', encoding='utf-8'))
    best_idx = int(meta['best_idx'])
    by_id = {}
    with open(os.path.join(debug_step_dir, 'candidates_world.csv'), newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['sample_id'])
            by_id.setdefault(sid, []).append((int(row['step']), float(row['x']), float(row['y'])))
    rows = sorted(by_id[best_idx], key=lambda x: x[0])
    return np.array([[x, y] for _, x, y in rows], dtype=np.float32), best_idx


def load_ego_pred(debug_step_dir):
    pts = []
    with open(os.path.join(debug_step_dir, 'ego_pred_world.csv'), newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append([float(row['x']), float(row['y'])])
    return np.array(pts, dtype=np.float32)


def load_adv_exec(trajectory_csv):
    pts = []
    with open(trajectory_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append([float(row['adv_x']), float(row['adv_y'])])
    return np.array(pts, dtype=np.float32)


def match_local_exec(selected, executed):
    if selected.size == 0 or executed.size == 0:
        return np.empty((0, 2), dtype=np.float32), 0
    start = selected[0]
    dist = np.linalg.norm(executed - start[None, :], axis=1)
    idx = int(np.argmin(dist))
    end_idx = min(len(executed), idx + len(selected))
    local = executed[idx:end_idx]
    if len(local) < len(selected) and len(local) > 0:
        pad = np.repeat(local[-1][None, :], len(selected) - len(local), axis=0)
        local = np.concatenate([local, pad], axis=0)
    return local, idx


def mean_abs_heading_change(points):
    if len(points) < 3:
        return 0.0
    headings = []
    for i in range(1, len(points)):
        dx = float(points[i, 0] - points[i - 1, 0])
        dy = float(points[i, 1] - points[i - 1, 1])
        if math.hypot(dx, dy) > 1e-6:
            headings.append(math.atan2(dy, dx))
    if len(headings) < 2:
        return 0.0
    turns = []
    for i in range(1, len(headings)):
        d = headings[i] - headings[i - 1]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        turns.append(abs(d))
    return float(sum(turns) / max(1, len(turns)))


def draw_panel(ax, title, selected, executed_local, ego_pred, tag):
    color_raw = '#D55E00'
    color_exec = '#0072B2'
    color_ego = '#444444'

    ax.plot(ego_pred[:, 0], ego_pred[:, 1], color=color_ego, linewidth=1.8, alpha=0.55, zorder=1)
    ax.plot(selected[:, 0], selected[:, 1], color=color_raw, linewidth=2.8, zorder=3)
    ax.plot(executed_local[:, 0], executed_local[:, 1], color=color_exec, linewidth=2.4, linestyle='--', zorder=4)

    ax.scatter(selected[0, 0], selected[0, 1], color='black', s=38, marker='o', zorder=5)
    ax.scatter(selected[-1, 0], selected[-1, 1], color=color_raw, s=52, marker='x', linewidths=2, zorder=5)
    ax.scatter(executed_local[-1, 0], executed_local[-1, 1], color=color_exec, s=46, marker='s', zorder=5)

    raw_turn = mean_abs_heading_change(selected)
    exec_turn = mean_abs_heading_change(executed_local)
    note = f'raw avg|Δθ|={raw_turn:.2f} rad\nexec avg|Δθ|={exec_turn:.2f} rad'
    ax.text(
        0.03, 0.97, note,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#BBBBBB', alpha=0.92),
    )
    ax.text(
        0.02, 0.05, tag,
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.9),
    )

    ax.set_title(title)
    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.grid(True, alpha=0.25)
    ax.axis('equal')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uncond-debug-step-dir', required=True)
    parser.add_argument('--uncond-trajectory-csv', required=True)
    parser.add_argument('--ours-debug-step-dir', required=True)
    parser.add_argument('--ours-trajectory-csv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 140})

    panels = [
        ('无视觉条件扩散', args.uncond_debug_step_dir, args.uncond_trajectory_csv, '1'),
        ('本文方法', args.ours_debug_step_dir, args.ours_trajectory_csv, '2'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    for ax, (title, debug_step_dir, traj_csv, tag) in zip(axes, panels):
        selected, _ = load_best_candidate(debug_step_dir)
        ego_pred = load_ego_pred(debug_step_dir)
        executed = load_adv_exec(traj_csv)
        executed_local, _ = match_local_exec(selected, executed)
        draw_panel(ax, title, selected, executed_local, ego_pred, tag)

    legend_items = [
        Line2D([0], [0], color='#D55E00', lw=2.8, label='raw 选中局部轨迹'),
        Line2D([0], [0], color='#0072B2', lw=2.4, linestyle='--', label='实际执行局部轨迹'),
        Line2D([0], [0], color='#444444', lw=1.8, alpha=0.55, label='Ego 预测轨迹'),
    ]
    fig.legend(handles=legend_items, loc='lower center', ncol=3, framealpha=0.92, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
