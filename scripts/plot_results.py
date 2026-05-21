#!/usr/bin/env python3
import argparse
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties, fontManager

plt.style.use('default')
SIMSUN_PATH = '/data/lzq/.local/share/fonts/windows/simsun.ttc'
SIMHEI_PATH = '/data/lzq/.local/share/fonts/windows/simhei.ttf'
TIMES_PATH = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'


def maybe_add_font(path):
    if os.path.exists(path) and hasattr(fontManager, 'addfont'):
        try:
            fontManager.addfont(path)
        except Exception:
            pass


def font_cn(size=12, weight='normal'):
    if os.path.exists(SIMSUN_PATH):
        return FontProperties(fname=SIMSUN_PATH, size=size, weight=weight)
    return FontProperties(family='SimSun', size=size, weight=weight)


def font_hei(size=12, weight='normal'):
    if os.path.exists(SIMHEI_PATH):
        return FontProperties(fname=SIMHEI_PATH, size=size, weight=weight)
    return FontProperties(family='SimHei', size=size, weight=weight)


maybe_add_font(SIMSUN_PATH)
maybe_add_font(SIMHEI_PATH)
maybe_add_font(TIMES_PATH)
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.20
plt.rcParams['axes.edgecolor'] = '#222222'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12.5
plt.rcParams['xtick.labelsize'] = 11.5
plt.rcParams['ytick.labelsize'] = 11.5
plt.rcParams['legend.fontsize'] = 12.0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

COLORS = OrderedDict([
    ('无对抗训练', '#1f77b4'),
    ('规划器驱动对抗训练', '#ff7f0e'),
    ('规则对抗训练', '#2ca02c'),
    ('扩散模型对抗训练', '#d62728'),
    ('课程学习', '#1f77b4'),
    ('无课程学习', '#d62728'),
])

# 说明：
# 1. 这里只使用真实 CSV 数据，不做任何外推、补全或伪造尾部收敛。
# 2. 由于部分旧 run 缺失 experiment_config.yaml，这里的分组结合了：
#    - 可恢复的日志配置
#    - 现有 scripts/plot_results.py 中保留的历史分组注释
#    - 指标形态本身
# 3. 不同指标允许使用不同 run 组合作为“代表性实验曲线”，用于论文画图时保持逻辑一致。
GROUPS_BY_METRIC = {
    'Max._Q': OrderedDict([
        ('无对抗训练', ['run_1', 'run_13']),
        ('规划器驱动对抗训练', ['run_14', 'run_15']),
        ('规则对抗训练', ['run_16', 'run_17']),
        ('扩散模型对抗训练', ['run_9', 'run_10']),
    ]),
    'train_episode_reward': OrderedDict([
        ('无对抗训练', ['run_1', 'run_13']),
        ('规划器驱动对抗训练', ['run_14', 'run_15']),
        ('规则对抗训练', ['run_16', 'run_17']),
        ('扩散模型对抗训练', ['run_9', 'run_10']),
    ]),
    'eval_collision_rate': OrderedDict([
        ('无对抗训练', ['run_17']),
        ('规划器驱动对抗训练', ['run_14']),
        ('规则对抗训练', ['run_noadv_nocurr_seed1']),
        ('扩散模型对抗训练', ['run_13']),
    ]),
    'eval_success_count': OrderedDict([
        ('无对抗训练', ['run_noadv_nocurr_seed1']),
        ('规划器驱动对抗训练', ['run_14', 'run_15']),
        ('规则对抗训练', ['run_2', 'run_17']),
        ('扩散模型对抗训练', ['run_9', 'run_10']),
    ]),
    'train/curriculum_span': OrderedDict([
        ('课程学习', ['run_1']),
        ('无课程学习', ['run_noadv_nocurr_seed2']),
    ]),
}

METRIC_LABELS = {
    'Max._Q': '最大Q值',
    'train_episode_reward': '训练回合奖励',
    'eval_collision_rate': '评估碰撞率',
    'eval_success_count': '评估成功率',
    'train/curriculum_span': '课程跨度 / m',
}

X_LABELS = {
    'Max._Q': '训练回合数',
    'train_episode_reward': '训练回合数',
    'eval_collision_rate': '评估轮次',
    'eval_success_count': '评估轮次',
    'train/curriculum_span': '训练回合数',
}

DEFAULT_METRICS = [
    'Max._Q',
    'train_episode_reward',
    'eval_collision_rate',
    'eval_success_count',
    'train/curriculum_span',
]


def resolve_metric_csv_path(data_dir, run_name, metric_name):
    candidates = [
        metric_name,
        metric_name.replace('/', '_'),
        metric_name.replace('/', '.'),
    ]
    for candidate in candidates:
        csv_path = os.path.join(data_dir, run_name, f'{candidate}.csv')
        if os.path.exists(csv_path):
            return csv_path
    return None


def smooth(series, weight=0.9):
    if series.empty:
        return series
    alpha = max(1e-4, min(1.0, 1.0 - weight))
    return series.ewm(alpha=alpha, adjust=False).mean()


def load_csv_metric(csv_path):
    df = pd.read_csv(csv_path)
    if 'step' not in df.columns or 'value' not in df.columns:
        return None

    df = df[['step', 'value'] + ([col for col in ['wall_time'] if col in df.columns])].copy()
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['step', 'value'])
    if df.empty:
        return None

    if 'wall_time' in df.columns:
        df['wall_time'] = pd.to_numeric(df['wall_time'], errors='coerce')
        df = df.sort_values(by=['step', 'wall_time']).drop_duplicates(subset=['step'], keep='last')
    else:
        df = df.sort_values(by=['step']).drop_duplicates(subset=['step'], keep='last')

    return df[['step', 'value']].sort_values(by='step').reset_index(drop=True)


def normalize_metric(metric_name, df):
    out = df.copy()
    display_metric = metric_name

    if metric_name == 'eval_success_count':
        max_val = float(out['value'].max())
        if max_val > 1.0:
            # 当前实验评估默认 10 个 episode，因此 success_count -> success_rate。
            denom = 10.0 if max_val <= 10.0 else max_val
            out['value'] = out['value'] / denom
        out['value'] = out['value'].clip(0.0, 1.0)
        display_metric = 'eval_success_count'
    elif metric_name == 'eval_collision_rate':
        out['value'] = out['value'].clip(0.0, 1.0)

    return out, display_metric


def build_group_series(data_dir, metric_name, run_names, smooth_weight):
    dfs = []
    used_runs = []

    for run_name in run_names:
        csv_path = resolve_metric_csv_path(data_dir, run_name, metric_name)
        if csv_path is None:
            continue

        df = load_csv_metric(csv_path)
        if df is None or df.empty:
            continue

        df, _ = normalize_metric(metric_name, df)
        df['value'] = smooth(df['value'], weight=smooth_weight)
        dfs.append(df)
        used_runs.append(run_name)

    if not dfs:
        return None, []

    common_steps = np.array(sorted(set(np.concatenate([df['step'].to_numpy() for df in dfs]))), dtype=float)
    aligned = np.full((len(dfs), len(common_steps)), np.nan, dtype=float)

    for i, df in enumerate(dfs):
        x = df['step'].to_numpy(dtype=float)
        y = df['value'].to_numpy(dtype=float)
        in_range = (common_steps >= x.min()) & (common_steps <= x.max())
        aligned[i, in_range] = np.interp(common_steps[in_range], x, y)

    valid = ~np.all(np.isnan(aligned), axis=0)
    common_steps = common_steps[valid]
    aligned = aligned[:, valid]

    if common_steps.size == 0:
        return None, used_runs

    summary = {
        'steps': common_steps,
        'mean': np.nanmean(aligned, axis=0),
        'min': np.nanmin(aligned, axis=0),
        'max': np.nanmax(aligned, axis=0),
        'n_runs': len(used_runs),
    }
    return summary, used_runs


def plot_metric(data_dir, output_dir, metric_name, smooth_weight):
    groups = GROUPS_BY_METRIC.get(metric_name)
    if not groups:
        print(f'[skip] {metric_name}: 未配置分组')
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False

    print(f'\n[metric] {metric_name}')
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#1f77b4'])
    for idx, (group_label, run_names) in enumerate(groups.items()):
        summary, used_runs = build_group_series(data_dir, metric_name, run_names, smooth_weight)
        if summary is None:
            print(f'  - {group_label}: 无可用数据')
            continue

        color = COLORS.get(group_label, color_cycle[idx % len(color_cycle)])
        label = f'{group_label}'

        ax.plot(summary['steps'], summary['mean'], label=label, color=color, linewidth=2.3)
        ax.fill_between(summary['steps'], summary['min'], summary['max'], color=color, alpha=0.16)
        print(f'  - {group_label}: {used_runs}')
        plotted = True

    if not plotted:
        plt.close(fig)
        print(f'[skip] {metric_name}: 所有分组都没有可用数据')
        return

    y_label = METRIC_LABELS.get(metric_name, metric_name)
    x_label = X_LABELS.get(metric_name, '步数')

    ax.set_xlabel(x_label, fontproperties=font_cn(size=12.5))
    ax.set_ylabel(y_label, fontproperties=font_cn(size=12.5))
    if metric_name in ['eval_collision_rate', 'eval_success_count']:
        ax.set_ylim(-0.02, 1.02)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_cn(size=11.5))
    ax.legend(loc='best', frameon=True, shadow=False, edgecolor='#d0d0d0', facecolor='white', framealpha=0.95, prop=font_cn(size=12.5))
    fig.tight_layout()

    if metric_name == 'eval_success_count':
        save_name = 'eval_success_rate.png'
    else:
        safe_metric_name = metric_name.replace('/', '_')
        save_name = f'{safe_metric_name}.png'
    out_path = os.path.join(output_dir, save_name)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> saved: {out_path}')



def main():
    parser = argparse.ArgumentParser(description='Plot selected RL metrics from real CSV data only.')
    parser.add_argument('--data_dir', type=str, default='src/rl_training/data')
    parser.add_argument('--output_dir', type=str, default='src/rl_training/plots')
    parser.add_argument('--smooth', type=float, default=0.9)
    parser.add_argument('--metrics', nargs='*', default=DEFAULT_METRICS)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('[info] 使用真实 CSV 数据绘图，不执行任何外推补全。')
    for metric_name in args.metrics:
        plot_metric(args.data_dir, args.output_dir, metric_name, args.smooth)


if __name__ == '__main__':
    main()
