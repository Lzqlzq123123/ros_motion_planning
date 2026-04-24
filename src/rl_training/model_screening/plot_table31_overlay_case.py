#!/usr/bin/env python3
import argparse
import csv
import os
import math

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


def load_traj(csv_path):
    ego_x, ego_y, adv_x, adv_y = [], [], [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ego_x.append(float(row['ego_x']))
            ego_y.append(float(row['ego_y']))
            adv_x.append(float(row['adv_x']))
            adv_y.append(float(row['adv_y']))
    return ego_x, ego_y, adv_x, adv_y


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rule-dir', required=True)
    parser.add_argument('--planner-dir', required=True)
    parser.add_argument('--uncond-dir', required=True)
    parser.add_argument('--ours-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ego-source', choices=['rule', 'planner', 'uncond', 'ours', 'longest'], default='longest')
    parser.add_argument('--paired-ego', action='store_true')
    parser.add_argument('--annotate-min-distance', action='store_true')
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update({
        'font.size': 11,
        'figure.dpi': 140,
    })

    configs = [
        ('规则对抗', args.rule_dir, '#E69F00', '-'),
        ('规划器驱动对抗', args.planner_dir, '#0072B2', '--'),
        ('无条件扩散对抗', args.uncond_dir, '#CC79A7', '-.'),
        ('本文方法', args.ours_dir, '#D55E00', '-'),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ego_candidates = {}
    adv_candidates = {}
    adv_start = None

    method_keys = ['rule', 'planner', 'uncond', 'ours']
    for method_key, (label, folder, color, linestyle) in zip(method_keys, configs):
        traj_csv = os.path.join(folder, 'trajectory.csv')
        ego_x, ego_y, adv_x, adv_y = load_traj(traj_csv)
        ego_candidates[method_key] = (ego_x, ego_y)
        adv_candidates[method_key] = (adv_x, adv_y)
        if adv_start is None:
            adv_start = (adv_x[0], adv_y[0])

        ax.plot(adv_x, adv_y, color=color, linestyle=linestyle, linewidth=2.2, label=label, zorder=3)
        ax.scatter(adv_x[0], adv_y[0], color=color, s=34, zorder=4)
        ax.scatter(adv_x[-1], adv_y[-1], color=color, s=60, marker='x', linewidths=2, zorder=5)

    def path_len(xy):
        xs, ys = xy
        return sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(xs)))

    if args.paired_ego:
        for method_key, (label, folder, color, linestyle) in zip(method_keys, configs):
            ego_x, ego_y = ego_candidates[method_key]
            ax.plot(ego_x, ego_y, color=color, linestyle=':', linewidth=2.0, alpha=0.95, zorder=2)
            if args.annotate_min_distance:
                adv_x, adv_y = adv_candidates[method_key]
                n = min(len(ego_x), len(adv_x))
                best_i = 0
                best_d = float('inf')
                for i in range(n):
                    d = math.hypot(ego_x[i] - adv_x[i], ego_y[i] - adv_y[i])
                    if d < best_d:
                        best_d = d
                        best_i = i
                mx = (ego_x[best_i] + adv_x[best_i]) / 2.0
                my = (ego_y[best_i] + adv_y[best_i]) / 2.0
                ax.plot([ego_x[best_i], adv_x[best_i]], [ego_y[best_i], adv_y[best_i]],
                        color=color, linewidth=1.4, alpha=0.75, zorder=4)
                ax.scatter([ego_x[best_i], adv_x[best_i]], [ego_y[best_i], adv_y[best_i]],
                           color=color, s=18, zorder=5)
                ax.annotate(f'{best_d:.2f} m', xy=(mx, my), xytext=(4, 4),
                            textcoords='offset points', fontsize=8, color=color)

        ref_key = 'ours' if 'ours' in ego_candidates else max(ego_candidates, key=lambda k: path_len(ego_candidates[k]))
        ego_x, ego_y = ego_candidates[ref_key]
        adv_x, adv_y = adv_candidates[ref_key]
        ax.scatter(ego_x[0], ego_y[0], color='black', s=46, marker='o', zorder=6)
        ax.scatter(ego_x[-1], ego_y[-1], color='black', s=60, marker='s', zorder=6)
        if adv_start is not None:
            ax.annotate('对抗车起点', xy=adv_start, xytext=(8, 8), textcoords='offset points', fontsize=9)
        ax.annotate('Ego 起点', xy=(ego_x[0], ego_y[0]), xytext=(8, -18), textcoords='offset points', fontsize=9)
        ax.annotate('Ego 终点', xy=(ego_x[-1], ego_y[-1]), xytext=(8, -18), textcoords='offset points', fontsize=9)

        method_handles = [
            Line2D([0], [0], color=color, lw=2.2, linestyle=linestyle, label=label)
            for label, folder, color, linestyle in configs
        ]
        style_handles = [
            Line2D([0], [0], color='black', lw=2.2, linestyle='-', label='对抗车轨迹'),
            Line2D([0], [0], color='black', lw=2.0, linestyle=':', label='对应 Ego 轨迹'),
        ]
        leg1 = ax.legend(handles=method_handles, loc='upper right', framealpha=0.92)
        ax.add_artist(leg1)
        ax.legend(handles=style_handles, loc='lower left', framealpha=0.92)
    else:
        if args.ego_source == 'longest':
            ego_key = max(ego_candidates, key=lambda k: path_len(ego_candidates[k]))
        else:
            ego_key = args.ego_source

        ego = ego_candidates[ego_key]
        ego_x, ego_y = ego
        ax.plot(ego_x, ego_y, color='black', linewidth=2.8, label=f'Ego 车轨迹', zorder=2)
        ax.scatter(ego_x[0], ego_y[0], color='black', s=46, marker='o', zorder=5)
        ax.scatter(ego_x[-1], ego_y[-1], color='black', s=60, marker='s', zorder=5)

        if adv_start is not None:
            ax.annotate('对抗车起点', xy=adv_start, xytext=(8, 8), textcoords='offset points', fontsize=9)
        ax.annotate('Ego 起点', xy=(ego_x[0], ego_y[0]), xytext=(8, -18), textcoords='offset points', fontsize=9)
        ax.annotate('Ego 终点', xy=(ego_x[-1], ego_y[-1]), xytext=(8, -18), textcoords='offset points', fontsize=9)
        ax.legend(loc='upper right', framealpha=0.92)

    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.grid(True, alpha=0.28)
    ax.axis('equal')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
