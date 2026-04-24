#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os

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


def load_ego(csv_path):
    xs, ys = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row['ego_x']))
            ys.append(float(row['ego_y']))
    return xs, ys


def load_adv_exec(csv_path):
    xs, ys = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'adv_x' in row and 'adv_y' in row:
                xs.append(float(row['adv_x']))
                ys.append(float(row['adv_y']))
            else:
                xs.append(float(row['x']))
                ys.append(float(row['y']))
    return xs, ys


def load_selected_nomad(csv_path, meta_path):
    meta = json.load(open(meta_path, 'r', encoding='utf-8'))
    best_idx = int(meta['best_idx'])
    xs, ys = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['sample_id']) == best_idx:
                xs.append(float(row['x']))
                ys.append(float(row['y']))
    if not xs:
        raise RuntimeError(f'No selected trajectory found in {csv_path} for best_idx={best_idx}')
    return xs, ys, best_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ego-csv', required=True)
    parser.add_argument('--rule-csv', required=True)
    parser.add_argument('--planner-csv', required=True)
    parser.add_argument('--shared-adv-start-csv', default=None)
    parser.add_argument('--shared-start-x', type=float, default=None)
    parser.add_argument('--shared-start-y', type=float, default=None)
    parser.add_argument('--uncond-candidates-csv', required=True)
    parser.add_argument('--uncond-meta-json', required=True)
    parser.add_argument('--ours-candidates-csv', required=True)
    parser.add_argument('--ours-meta-json', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    setup_fonts()
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 140})

    ego_x, ego_y = load_ego(args.ego_csv)
    rule_x, rule_y = load_adv_exec(args.rule_csv)
    planner_x, planner_y = load_adv_exec(args.planner_csv)
    uncond_x, uncond_y, uncond_idx = load_selected_nomad(args.uncond_candidates_csv, args.uncond_meta_json)
    ours_x, ours_y, ours_idx = load_selected_nomad(args.ours_candidates_csv, args.ours_meta_json)

    if args.shared_start_x is not None and args.shared_start_y is not None:
        sx, sy = float(args.shared_start_x), float(args.shared_start_y)
        if math.hypot(planner_x[0] - sx, planner_y[0] - sy) > 1e-6:
            planner_x = [sx] + planner_x
            planner_y = [sy] + planner_y
        if math.hypot(uncond_x[0] - sx, uncond_y[0] - sy) > 1e-6:
            uncond_x = [sx] + uncond_x
            uncond_y = [sy] + uncond_y
        if math.hypot(ours_x[0] - sx, ours_y[0] - sy) > 1e-6:
            ours_x = [sx] + ours_x
            ours_y = [sy] + ours_y
    elif args.shared_adv_start_csv:
        shared_x, shared_y = load_adv_exec(args.shared_adv_start_csv)
        if shared_x:
            sx, sy = shared_x[0], shared_y[0]
            if math.hypot(planner_x[0] - sx, planner_y[0] - sy) > 1e-6:
                planner_x = [sx] + planner_x
                planner_y = [sy] + planner_y
            if math.hypot(uncond_x[0] - sx, uncond_y[0] - sy) > 1e-6:
                uncond_x = [sx] + uncond_x
                uncond_y = [sy] + uncond_y
            if math.hypot(ours_x[0] - sx, ours_y[0] - sy) > 1e-6:
                ours_x = [sx] + ours_x
                ours_y = [sy] + ours_y

    fig, ax = plt.subplots(figsize=(8.2, 6.7))

    all_x = ego_x + rule_x + planner_x + uncond_x + ours_x
    all_y = ego_y + rule_y + planner_y + uncond_y + ours_y

    ax.plot(rule_x, rule_y, color='#E69F00', linewidth=2.3, zorder=3, label='规则对抗')
    ax.plot(planner_x, planner_y, color='#0072B2', linewidth=2.3, zorder=3, label='规划器驱动对抗')
    ax.plot(uncond_x, uncond_y, color='#CC79A7', linewidth=2.2, marker='o', markersize=2.8, zorder=4, label='无视觉条件扩散对抗')
    ax.plot(ours_x, ours_y, color='#D55E00', linewidth=2.2, marker='o', markersize=2.8, zorder=4, label='扩散模型对抗（本文）')
    ax.plot(ego_x, ego_y, color='black', linewidth=2.8, zorder=2, label='Ego 车轨迹')

    starts = [(rule_x[0], rule_y[0]), (planner_x[0], planner_y[0]), (uncond_x[0], uncond_y[0]), (ours_x[0], ours_y[0])]
    max_start_dist = 0.0
    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            max_start_dist = max(max_start_dist, math.hypot(starts[i][0] - starts[j][0], starts[i][1] - starts[j][1]))
    if max_start_dist < 0.12:
        sx = sum(p[0] for p in starts) / len(starts)
        sy = sum(p[1] for p in starts) / len(starts)
        ax.scatter(sx, sy, s=66, facecolors='white', edgecolors='black', linewidths=1.3, zorder=5)
        ax.annotate('对抗车共同起点', xy=(sx, sy), xytext=(8, 8), textcoords='offset points', fontsize=9)
    else:
        for sx, sy in starts:
            ax.scatter(sx, sy, s=26, color='black', zorder=5)

    for x, y, c in [(rule_x[-1], rule_y[-1], '#E69F00'), (planner_x[-1], planner_y[-1], '#0072B2'), (uncond_x[-1], uncond_y[-1], '#CC79A7'), (ours_x[-1], ours_y[-1], '#D55E00')]:
        ax.scatter(x, y, marker='x', s=58, linewidths=2, color=c, zorder=5)
    ax.scatter(ego_x[0], ego_y[0], color='black', s=34, zorder=5)
    ax.scatter(ego_x[-1], ego_y[-1], color='black', marker='s', s=48, zorder=5)

    ax.annotate('Ego 起点', xy=(ego_x[0], ego_y[0]), xytext=(8, -16), textcoords='offset points', fontsize=9)
    ax.annotate('Ego 终点', xy=(ego_x[-1], ego_y[-1]), xytext=(8, -16), textcoords='offset points', fontsize=9)

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

    handles = [
        Line2D([0], [0], color='#E69F00', lw=2.3, label='规则对抗'),
        Line2D([0], [0], color='#0072B2', lw=2.3, label='规划器驱动对抗'),
        Line2D([0], [0], color='#CC79A7', lw=2.2, marker='o', markersize=4, label='无视觉条件扩散对抗（原始选中）'),
        Line2D([0], [0], color='#D55E00', lw=2.2, marker='o', markersize=4, label='扩散模型对抗（本文，原始选中）'),
        Line2D([0], [0], color='black', lw=2.8, label='Ego 车轨迹'),
    ]
    ax.legend(handles=handles, loc='upper right', framealpha=0.93)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=320, bbox_inches='tight')
    print(f'Saved to {args.output}; uncond_best_idx={uncond_idx}, ours_best_idx={ours_idx}')


if __name__ == '__main__':
    main()
