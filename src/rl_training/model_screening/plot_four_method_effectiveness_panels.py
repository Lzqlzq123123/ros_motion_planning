#!/usr/bin/env python3
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager


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


def load_traj(csv_path):
    ego_x, ego_y, adv_x, adv_y = [], [], [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ego_x.append(float(row['ego_x']))
            ego_y.append(float(row['ego_y']))
            adv_x.append(float(row['adv_x']))
            adv_y.append(float(row['adv_y']))
    return ego_x, ego_y, adv_x, adv_y


def min_distance(ego_x, ego_y, adv_x, adv_y):
    n = min(len(ego_x), len(adv_x))
    best_i = 0
    best_d = float('inf')
    for i in range(n):
        d = math.hypot(ego_x[i] - adv_x[i], ego_y[i] - adv_y[i])
        if d < best_d:
            best_d = d
            best_i = i
    return best_d, best_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rule-csv', required=True)
    parser.add_argument('--planner-csv', required=True)
    parser.add_argument('--uncond-csv', required=True)
    parser.add_argument('--ours-csv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 140})

    configs = [
        ('规则对抗', args.rule_csv, '#E69F00'),
        ('规划器驱动对抗', args.planner_csv, '#0072B2'),
        ('无视觉条件扩散对抗', args.uncond_csv, '#CC79A7'),
        ('扩散模型对抗（本文）', args.ours_csv, '#D55E00'),
    ]

    data = []
    all_x = []
    all_y = []
    for title, csv_path, color in configs:
        ego_x, ego_y, adv_x, adv_y = load_traj(csv_path)
        data.append((title, color, ego_x, ego_y, adv_x, adv_y))
        all_x.extend(ego_x + adv_x)
        all_y.extend(ego_y + adv_y)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = max(0.15, 0.08 * (x_max - x_min))
    y_pad = max(0.15, 0.08 * (y_max - y_min))

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    axes = axes.ravel()

    for idx, (ax, item) in enumerate(zip(axes, data), start=1):
        title, color, ego_x, ego_y, adv_x, adv_y = item
        dmin, best_i = min_distance(ego_x, ego_y, adv_x, adv_y)

        ax.plot(adv_x, adv_y, color=color, linewidth=2.6, zorder=3)
        ax.plot(ego_x, ego_y, color='black', linewidth=2.2, linestyle='--', zorder=2)

        ax.scatter(adv_x[0], adv_y[0], color=color, s=34, zorder=4)
        ax.scatter(adv_x[-1], adv_y[-1], color=color, marker='x', s=56, linewidths=2, zorder=4)
        ax.scatter(ego_x[0], ego_y[0], color='black', s=34, zorder=4)
        ax.scatter(ego_x[-1], ego_y[-1], color='black', marker='s', s=42, zorder=4)

        ax.plot([adv_x[best_i], ego_x[best_i]], [adv_y[best_i], ego_y[best_i]],
                color=color, linewidth=1.4, alpha=0.8, zorder=4)

        ax.text(
            0.03, 0.96,
            f'{idx}  {title}\n最小距离 = {dmin:.2f} m',
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#BBBBBB', alpha=0.94),
        )

        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.25)
        ax.set_xlabel('X / m')
        ax.set_ylabel('Y / m')

    handles = [
        plt.Line2D([0], [0], color='#666666', lw=2.6, label='对抗车轨迹'),
        plt.Line2D([0], [0], color='black', lw=2.2, linestyle='--', label='Ego 车轨迹'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, framealpha=0.94, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=320, bbox_inches='tight')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
