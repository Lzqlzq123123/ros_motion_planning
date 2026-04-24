#!/usr/bin/env python3
import argparse
import csv
import json
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


def load_candidates(path):
    by_id = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row['sample_id'])
            by_id.setdefault(sid, []).append((int(row['step']), float(row['x']), float(row['y'])))
    trajs = []
    for sid in sorted(by_id):
        rows = sorted(by_id[sid], key=lambda x: x[0])
        trajs.append(np.array([[x, y] for _, x, y in rows], dtype=np.float32))
    return trajs


def load_ego(path):
    pts = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append([float(row['x']), float(row['y'])])
    return np.array(pts, dtype=np.float32)


def smooth_polyline(points: np.ndarray, samples_per_seg: int = 18) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= 2:
        return points

    tangents = np.zeros_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangents[1:-1] = 0.5 * (points[2:] - points[:-2])

    smooth_pts = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        m0 = tangents[i]
        m1 = tangents[i + 1]
        ts = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False, dtype=np.float32)
        for t in ts:
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            smooth_pts.append(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)
    smooth_pts.append(points[-1])
    return np.asarray(smooth_pts, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--title', default='')
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 120})

    candidates = load_candidates(os.path.join(args.debug_dir, 'candidates_world.csv'))
    ego = load_ego(os.path.join(args.debug_dir, 'ego_pred_world.csv'))
    meta = json.load(open(os.path.join(args.debug_dir, 'selection_meta.json'), 'r', encoding='utf-8'))
    best_idx = int(meta['best_idx'])

    fig, ax = plt.subplots(figsize=(6.8, 6.3))
    normal_color = '#4C9BE8'
    best_color = '#D55E00'

    for i, traj in enumerate(candidates):
        if i == best_idx:
            continue
        traj_s = smooth_polyline(traj)
        ax.plot(traj_s[:, 0], traj_s[:, 1], color=normal_color, alpha=0.22, linewidth=1.2)

    best = candidates[best_idx]
    best_s = smooth_polyline(best)
    ego_s = smooth_polyline(ego)
    ax.plot(best_s[:, 0], best_s[:, 1], color=best_color, linewidth=2.8, zorder=7)
    ax.plot(ego_s[:, 0], ego_s[:, 1], color='black', linewidth=2.5, zorder=6)

    ax.scatter(best[0, 0], best[0, 1], color='green', marker='*', s=130, zorder=8)
    ax.scatter(
        best[-1, 0], best[-1, 1],
        color=best_color, marker='X', s=110,
        edgecolors='white', linewidths=0.8, zorder=9,
    )
    ax.scatter(
        ego[-1, 0], ego[-1, 1],
        color='black', marker='s', s=70,
        edgecolors='white', linewidths=0.8, zorder=9,
    )

    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.grid(True, alpha=0.25)
    ax.axis('equal')
    if args.title:
        ax.set_title(args.title)

    legend_items = [
        Line2D([0], [0], color=normal_color, lw=1.2, alpha=0.3, label='候选轨迹'),
        Line2D([0], [0], color=best_color, lw=2.8, label='最高风险轨迹'),
        Line2D([0], [0], color='black', lw=2.5, label='Ego 预测轨迹'),
        Line2D([0], [0], color='green', marker='*', markersize=10, linewidth=0, label='起点'),
        Line2D([0], [0], color=best_color, marker='X', markersize=8, linewidth=0, label='筛选轨迹终点'),
        Line2D([0], [0], color='black', marker='s', markersize=7, linewidth=0, label='Ego终点'),
    ]
    ax.legend(
        handles=legend_items,
        loc='lower left',
        ncol=2,
        framealpha=0.92,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
