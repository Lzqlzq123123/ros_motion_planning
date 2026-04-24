#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path('/data/lzq/ros_motion_planning/src/rl_training')
OUT = ROOT / 'model_screening' / 'paper_results_20260320'
RAW = OUT / 'raw_csv'


def round1(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))


def pct(x: float) -> str:
    return f'{round1(x * 100):.1f}'


def read_episode_summary(csv_path: Path) -> dict:
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty csv: {csv_path}')
    for row in reversed(rows):
        if row.get('row_type') == 'summary':
            return row
    raise ValueError(f'No summary row found in {csv_path}')


def read_aggregate_row(csv_path: Path, *, scenario: str, key_field: str, key_value: str) -> dict:
    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get('scenario') == scenario and row.get(key_field) == key_value:
            return row
    raise ValueError(f'No row matched in {csv_path}: scenario={scenario}, {key_field}={key_value}')


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def training_dir_from_model_path(model_path: str) -> str:
    p = Path(model_path)
    if p.name == 'td3_model':
        return str(p.parent)
    return str(p)


OUT.mkdir(parents=True, exist_ok=True)
ensure_clean_dir(RAW)

# --- current final Table 4-3 sources ---
base_noadv_src = ROOT / 'eval_csv' / '原始无对抗.csv'
base_adv_src = ROOT / 'eval_csv' / '原始有对抗.csv'
rule_noadv_src = ROOT / 'model_screening' / 'independent_eval_batch_20260320' / 'csv' / 'rule_adversarial__forklift_movebase__run_10__noadv.csv'
rule_adv_src = ROOT / 'model_screening' / 'independent_eval_batch_20260320' / 'csv' / 'rule_adversarial__forklift_movebase__run_10__adv.csv'
planner_noadv_src = ROOT / 'model_screening' / 'independent_eval_batch_20260320' / 'csv' / 'planner_adversarial__forklift_movebase__run_18__noadv.csv'
planner_adv_src = ROOT / 'model_screening' / 'independent_eval_batch_20260320' / 'csv' / 'planner_adversarial__forklift_movebase__run_18__adv.csv'
ours_noadv_agg_src = ROOT / 'model_screening' / 'ours_repeat_eval_20260320' / 'repeat_eval_aggregate.csv'
ours_adv_agg_src = ROOT / 'model_screening' / 'ours_run1_adv_native_eval_20260320' / 'repeat_eval_aggregate.csv'

ours_noadv_repeat_srcs = [
    ROOT / 'model_screening' / 'ours_repeat_eval_20260320' / 'csv' / f'run_1__noadv__repeat{i}.csv'
    for i in (1, 2, 3)
]
ours_adv_repeat_srcs = [
    ROOT / 'model_screening' / 'ours_run1_adv_native_eval_20260320' / 'csv' / f'ours_run1__adv__repeat{i}.csv'
    for i in (1, 2, 3)
]

# copy exact source files used by current thesis tables
copy_plan = {
    'baseline_noadv_historical.csv': base_noadv_src,
    'baseline_adv_historical.csv': base_adv_src,
    'rule_run10_noadv_independent.csv': rule_noadv_src,
    'rule_run10_adv_independent.csv': rule_adv_src,
    'planner_run18_noadv_independent.csv': planner_noadv_src,
    'planner_run18_adv_independent.csv': planner_adv_src,
    'ours_run1_noadv_repeat_aggregate.csv': ours_noadv_agg_src,
    'ours_run1_adv_native_repeat_aggregate.csv': ours_adv_agg_src,
}
for i, src in enumerate(ours_noadv_repeat_srcs, 1):
    copy_plan[f'ours_run1_noadv_repeat{i}.csv'] = src
for i, src in enumerate(ours_adv_repeat_srcs, 1):
    copy_plan[f'ours_run1_adv_repeat{i}.csv'] = src

for name, src in copy_plan.items():
    copy_file(src, RAW / name)

# parse source data
base_noadv = read_episode_summary(base_noadv_src)
base_adv = read_episode_summary(base_adv_src)
rule_noadv = read_episode_summary(rule_noadv_src)
rule_adv = read_episode_summary(rule_adv_src)
planner_noadv = read_episode_summary(planner_noadv_src)
planner_adv = read_episode_summary(planner_adv_src)
ours_noadv = read_aggregate_row(ours_noadv_agg_src, scenario='noadv', key_field='model_name', key_value='run_1')
ours_adv = read_aggregate_row(ours_adv_agg_src, scenario='adv', key_field='model_key', key_value='ours_run1')

methods = [
    {
        'training_method': '历史参考对照 TD3',
        'display_label': '历史参考对照 TD3',
        'training_log_dir': '/data/lzq/ros_motion_planning/src/rl_training/logs/forklift_movebase/run_19',
        'model_path': 'logs/forklift_movebase/run_19/td3_model',
        'noadv': {
            'source_type': '历史 eval_csv 汇总行',
            'source_csv': str(RAW / 'baseline_noadv_historical.csv'),
            'success_rate': float(base_noadv['success_rate']),
            'collision_rate': float(base_noadv['collision_rate']),
            'avg_reward': float(base_noadv['avg_reward']),
            'notes': '仅作历史参考对照，不再严格表述为纯无对抗训练基线',
        },
        'adv': {
            'source_type': '历史 eval_csv 汇总行',
            'source_csv': str(RAW / 'baseline_adv_historical.csv'),
            'success_rate': float(base_adv['success_rate']),
            'collision_rate': float(base_adv['collision_rate']),
            'avg_reward': float(base_adv['avg_reward']),
            'notes': '仅作历史参考对照，不再严格表述为纯无对抗训练基线',
        },
    },
    {
        'training_method': '规则对抗训练 TD3',
        'display_label': '规则对抗训练 TD3',
        'training_log_dir': '/data/lzq/ros_motion_planning/src/rl_training/logs/forklift_movebase/run_10',
        'model_path': '/data/lzq/ros_motion_planning/src/rl_training/logs/forklift_movebase/run_10/td3_model',
        'noadv': {
            'source_type': '独立验证单次 CSV 汇总行',
            'source_csv': str(RAW / 'rule_run10_noadv_independent.csv'),
            'success_rate': float(rule_noadv['success_rate']),
            'collision_rate': float(rule_noadv['collision_rate']),
            'avg_reward': float(rule_noadv['avg_reward']),
            'notes': '论文表4-3当前采用的代表结果',
        },
        'adv': {
            'source_type': '独立验证单次 CSV 汇总行',
            'source_csv': str(RAW / 'rule_run10_adv_independent.csv'),
            'success_rate': float(rule_adv['success_rate']),
            'collision_rate': float(rule_adv['collision_rate']),
            'avg_reward': float(rule_adv['avg_reward']),
            'notes': '论文表4-3当前采用的代表结果',
        },
    },
    {
        'training_method': '规划器驱动对抗训练 TD3',
        'display_label': '规划器驱动对抗训练 TD3',
        'training_log_dir': '/data/lzq/ros_motion_planning/src/rl_training/logs/forklift_movebase/run_18',
        'model_path': '/data/lzq/ros_motion_planning/src/rl_training/logs/forklift_movebase/run_18/td3_model',
        'noadv': {
            'source_type': '独立验证单次 CSV 汇总行',
            'source_csv': str(RAW / 'planner_run18_noadv_independent.csv'),
            'success_rate': float(planner_noadv['success_rate']),
            'collision_rate': float(planner_noadv['collision_rate']),
            'avg_reward': float(planner_noadv['avg_reward']),
            'notes': '论文表4-3当前采用的代表结果',
        },
        'adv': {
            'source_type': '独立验证单次 CSV 汇总行',
            'source_csv': str(RAW / 'planner_run18_adv_independent.csv'),
            'success_rate': float(planner_adv['success_rate']),
            'collision_rate': float(planner_adv['collision_rate']),
            'avg_reward': float(planner_adv['avg_reward']),
            'notes': '论文表4-3当前采用的代表结果',
        },
    },
    {
        'training_method': '扩散模型对抗训练 TD3（本文，run_1）',
        'display_label': '扩散模型对抗训练 TD3（本文，run_1）',
        'training_log_dir': '/data/lzq/ros_motion_planning/src/rl_training/logs/forklift_movebase_with_goal/run_1',
        'model_path': '/data/lzq/ros_motion_planning/src/rl_training/logs/forklift_movebase_with_goal/run_1/td3_model',
        'noadv': {
            'source_type': '3次独立验证平均（repeat_eval_aggregate）',
            'source_csv': str(RAW / 'ours_run1_noadv_repeat_aggregate.csv'),
            'success_rate': float(ours_noadv['avg_success_rate']),
            'collision_rate': float(ours_noadv['avg_collision_rate']),
            'avg_reward': float(ours_noadv['avg_reward']),
            'notes': '对应原始重复验证 CSV: ours_run1_noadv_repeat1~3.csv',
        },
        'adv': {
            'source_type': '3次独立验证平均（native-config repeat_eval_aggregate）',
            'source_csv': str(RAW / 'ours_run1_adv_native_repeat_aggregate.csv'),
            'success_rate': float(ours_adv['avg_success_rate']),
            'collision_rate': float(ours_adv['avg_collision_rate']),
            'avg_reward': float(ours_adv['avg_reward']),
            'notes': '对应原始重复验证 CSV: ours_run1_adv_repeat1~3.csv；采用模型自身训练环境参数重测',
        },
    },
]

# wide summary used directly by table 4-3
rows_43 = [[
    '训练方式',
    '无对抗验证成功率（%）',
    '有对抗验证成功率（%）',
    '成功率波动幅度（|ΔSR|）',
    '无对抗结果来源',
    '有对抗结果来源',
    '训练模型路径',
]]
for m in methods:
    noadv_pct = round1(m['noadv']['success_rate'] * 100)
    adv_pct = round1(m['adv']['success_rate'] * 100)
    rows_43.append([
        m['display_label'],
        f'{noadv_pct:.1f}',
        f'{adv_pct:.1f}',
        f'{round1(abs(noadv_pct - adv_pct)):.1f}',
        Path(m['noadv']['source_csv']).name,
        Path(m['adv']['source_csv']).name,
        m['model_path'],
    ])
with (OUT / 'table_4_3_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
    csv.writer(f).writerows(rows_43)

# long-form traceability sheet for checking later
trace_rows = [[
    '训练方式', '验证场景', '成功率（%）', '碰撞率（%）', '平均奖励', '结果来源类型',
    '源文件', '训练日志目录', '模型路径', '备注'
]]
for m in methods:
    for scenario_key, scenario_label in [('noadv', '无对抗验证'), ('adv', '有对抗验证')]:
        item = m[scenario_key]
        trace_rows.append([
            m['display_label'],
            scenario_label,
            f"{round1(item['success_rate'] * 100):.1f}",
            f"{round1(item['collision_rate'] * 100):.1f}",
            f"{round1(item['avg_reward']):.1f}",
            item['source_type'],
            item['source_csv'],
            m['training_log_dir'],
            m['model_path'],
            item['notes'],
        ])
with (OUT / 'table_4_3_traceability.csv').open('w', encoding='utf-8-sig', newline='') as f:
    csv.writer(f).writerows(trace_rows)

# keep table 4-4 synchronized with current thesis values
ours_method = methods[-1]
rows_44 = [
    ['指标', '有对抗训练（本文，run_1）', '历史参考对照项', '结果来源'],
    ['评估平均奖励', f"{round1(ours_method['adv']['avg_reward']):.1f}", f"{round1(methods[0]['adv']['avg_reward']):.1f}", '本文：ours_run1_adv_native_repeat_aggregate.csv；对照：baseline_adv_historical.csv'],
    ['碰撞率', f"{round1(ours_method['adv']['collision_rate'] * 100):.1f}%", f"{round1(methods[0]['adv']['collision_rate'] * 100):.1f}%", '本文：ours_run1_adv_native_repeat_aggregate.csv；对照：baseline_adv_historical.csv'],
    ['任务成功次数（/10）', f"{round1(ours_method['adv']['success_rate'] * 10):.1f}", f"{round1(methods[0]['adv']['success_rate'] * 10):.1f}", '本文：ours_run1_adv_native_repeat_aggregate.csv；对照：baseline_adv_historical.csv'],
]
with (OUT / 'table_4_4_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
    csv.writer(f).writerows(rows_44)

manifest = {
    'generated_at': '2026-03-20',
    'purpose': '用于论文表4-3/表4-4当前最终版本的结果追溯与核查',
    'table_4_3_final': [
        {
            'training_method': m['display_label'],
            'noadv_success_percent': round1(m['noadv']['success_rate'] * 100),
            'adv_success_percent': round1(m['adv']['success_rate'] * 100),
            'delta_success_percent': round1(abs(m['noadv']['success_rate'] - m['adv']['success_rate']) * 100),
            'training_log_dir': m['training_log_dir'],
            'model_path': m['model_path'],
            'noadv_source_csv': m['noadv']['source_csv'],
            'adv_source_csv': m['adv']['source_csv'],
            'notes': {
                'noadv': m['noadv']['notes'],
                'adv': m['adv']['notes'],
            },
        }
        for m in methods
    ],
    'table_4_4_final': {
        'ours_run1_avg_reward': round1(ours_method['adv']['avg_reward']),
        'ours_run1_collision_percent': round1(ours_method['adv']['collision_rate'] * 100),
        'ours_run1_success_per10': round1(ours_method['adv']['success_rate'] * 10),
        'historical_ref_avg_reward': round1(methods[0]['adv']['avg_reward']),
        'historical_ref_collision_percent': round1(methods[0]['adv']['collision_rate'] * 100),
        'historical_ref_success_per10': round1(methods[0]['adv']['success_rate'] * 10),
    },
    'copied_raw_csv': {name: str(RAW / name) for name in sorted(copy_plan)},
}
with (OUT / 'paper_results_manifest.json').open('w', encoding='utf-8') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

readme = f'''# 论文表 4-3 / 表 4-4 对应 CSV 整理包（当前最终版）

生成时间：2026-03-20

## 本次整理目的

将论文**当前最终采用**的表 4-3、表 4-4 数值，与其对应的源 CSV、训练模型目录、重复验证结果统一整理，便于后续核查。

## 目录说明

- `raw_csv/`：当前最终表格实际采用的源 CSV 拷贝
- `table_4_3_summary.csv`：表 4-3 当前最终汇总（宽表）
- `table_4_3_traceability.csv`：表 4-3 逐项追溯表，含源文件与模型路径
- `table_4_4_summary.csv`：表 4-4 当前最终汇总
- `paper_results_manifest.json`：机器可读的结果追溯清单

## 表 4-3 当前最终采用

- 历史参考对照 TD3：72.8 -> 50.0，波动 22.8
- 规则对抗训练 TD3：84.2 -> 78.9，波动 5.3
- 规划器驱动对抗训练 TD3：89.5 -> 83.3，波动 6.2
- 扩散模型对抗训练 TD3（本文，run_1）：90.8 -> 80.5，波动 10.3

## 表 4-4 当前最终采用

- 评估平均奖励：37.1 vs -37.8
- 碰撞率：19.5% vs 47.1%
- 任务成功次数（/10）：8.1 vs 5.0

## 特别说明

1. “历史参考对照 TD3”来自旧版 `eval_csv/原始无对抗.csv` 与 `eval_csv/原始有对抗.csv`，当前仅作为历史参考对照项；
2. 本文方法不再采用“2 seeds 平均”作为主表结果，当前统一使用 `forklift_movebase_with_goal/run_1`；
3. 其中本文方法有对抗结果采用 `ours_run1_adv_native_repeat_aggregate.csv`，是基于模型自身训练环境参数进行的 3 次独立重测均值；
4. 本文方法无对抗结果采用 `ours_run1_noadv_repeat_aggregate.csv`，对应 3 次独立验证均值。
'''
(OUT / 'README.md').write_text(readme, encoding='utf-8')

print(f'Updated current paper result package at: {OUT}')
