#!/usr/bin/env python3
import argparse
import csv
import os
import math

import matplotlib.pyplot as plt
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ego-csv', required=True)
    parser.add_argument('--rule-csv', required=True)
    parser.add_argument('--planner-csv', required=True)
    parser.add_argument('--uncond-csv', required=True)
    parser.add_argument('--ours-csv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 140})

    ego_x, ego_y, _, _ = load_traj(args.ego_csv)
    configs = [
        ('规则对抗', args.rule_csv, '#E69F00'),
        ('规划器驱动对抗', args.planner_csv, '#0072B2'),
        ('无视觉条件扩散对抗', args.uncond_csv, '#CC79A7'),
        ('扩散模型对抗（本文）', args.ours_csv, '#D55E00'),
    ]

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    all_x = list(ego_x)
    all_y = list(ego_y)
    adv_starts = []

    for label, csv_path, color in configs:
        _, _, adv_x, adv_y = load_traj(csv_path)
        all_x.extend(adv_x)
        all_y.extend(adv_y)
        ax.plot(adv_x, adv_y, color=color, linewidth=2.3, label=label, zorder=3)
        adv_starts.append((adv_x[0], adv_y[0]))
        ax.scatter(adv_x[-1], adv_y[-1], color=color, marker='x', s=52, linewidths=2, zorder=4)

    shared_start = False
    if adv_starts:
        max_start_dist = 0.0
        for i in range(len(adv_starts)):
            for j in range(i + 1, len(adv_starts)):
                d = math.hypot(adv_starts[i][0] - adv_starts[j][0], adv_starts[i][1] - adv_starts[j][1])
                max_start_dist = max(max_start_dist, d)
        shared_start = max_start_dist < 0.12

    if shared_start:
        sx = sum(p[0] for p in adv_starts) / len(adv_starts)
        sy = sum(p[1] for p in adv_starts) / len(adv_starts)
        ax.scatter(sx, sy, s=62, facecolors='white', edgecolors='black', linewidths=1.3, zorder=5)
        ax.annotate('对抗车共同起点', xy=(sx, sy), xytext=(8, 8), textcoords='offset points', fontsize=9)
    else:
        for (sx, sy), (_, _, color) in zip(adv_starts, [(c[0], c[1], c[2]) for c in configs]):
            ax.scatter(sx, sy, color=color, s=28, zorder=4)

    ax.plot(ego_x, ego_y, color='black', linewidth=2.8, zorder=2, label='Ego 车轨迹')
    ax.scatter(ego_x[0], ego_y[0], color='black', s=34, zorder=4)
    ax.scatter(ego_x[-1], ego_y[-1], color='black', marker='s', s=48, zorder=4)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = max(0.18, 0.08 * (x_max - x_min))
    y_pad = max(0.18, 0.08 * (y_max - y_min))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.grid(True, alpha=0.25)
    ax.axis('equal')

    ax.annotate('Ego 起点', xy=(ego_x[0], ego_y[0]), xytext=(8, -16), textcoords='offset points', fontsize=9)
    ax.annotate('Ego 终点', xy=(ego_x[-1], ego_y[-1]), xytext=(8, -16), textcoords='offset points', fontsize=9)

    legend_handles = [
        Line2D([0], [0], color='#E69F00', lw=2.3, label='规则对抗'),
        Line2D([0], [0], color='#0072B2', lw=2.3, label='规划器驱动对抗'),
        Line2D([0], [0], color='#CC79A7', lw=2.3, label='无视觉条件扩散对抗'),
        Line2D([0], [0], color='#D55E00', lw=2.3, label='扩散模型对抗（本文）'),
        Line2D([0], [0], color='black', lw=2.8, label='Ego 车轨迹'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.93)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=320, bbox_inches='tight')
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
