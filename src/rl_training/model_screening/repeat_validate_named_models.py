#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path('/data/lzq/ros_motion_planning')
BASE_CONFIG = PROJECT_ROOT / 'src/rl_training/config/forklift_movebase.yaml'
EVAL_SCRIPT = PROJECT_ROOT / 'src/rl_training/eval_velodyne_td3_with_goal.py'
MAIN_SCRIPT = PROJECT_ROOT / 'scripts/main.sh'
KILL_SCRIPT = PROJECT_ROOT / 'scripts/killpro.sh'

MODEL_REGISTRY = {
    'noadv_seed1': {
        'label': '无对抗训练 TD3',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_noadv_nocurr_seed1/td3_model',
    },
    'noadv_seed2': {
        'label': '无对抗训练 TD3',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_noadv_nocurr_seed2/td3_model',
    },
    'noadv_run1': {
        'label': '无对抗训练 TD3',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_1/td3_model',
    },
    'rule_run10': {
        'label': '规则对抗训练 TD3',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_10/td3_model',
    },
    'planner_run18': {
        'label': '规划器驱动对抗训练 TD3',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_18/td3_model',
    },
    'ours_run1': {
        'label': '扩散模型对抗训练 TD3（本文）',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase_with_goal/run_1/td3_model',
    },
    'ours_run2': {
        'label': '扩散模型对抗训练 TD3（本文）',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase_with_goal/run_2/td3_model',
    },
}


def write_yaml(path: Path, data):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def build_eval_config(base_cfg, enabled: bool, mode: str = 'movebase'):
    cfg = json.loads(json.dumps(base_cfg))
    env = cfg['env']
    opponent = dict(env.get('opponent', {}))
    opponent['enabled'] = bool(enabled)
    opponent['mode'] = mode
    env['opponent'] = opponent
    return cfg


def parse_summary(csv_path: Path):
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        last_summary = None
        for row in reader:
            if row.get('row_type') == 'summary':
                last_summary = row
        return last_summary


def dump_json(path: Path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _run_bash(command: str, check=False):
    return subprocess.run(
        ['bash', '-lc', command],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=check,
    )


def stack_ready():
    probe = _run_bash(
        "source /opt/ros/noetic/setup.bash && "
        "rosservice list 2>/dev/null && "
        "echo '---TOPICS---' && "
        "rostopic list 2>/dev/null"
    )
    if probe.returncode != 0:
        return False
    text = probe.stdout
    required = [
        '/gazebo/set_model_state',
        '/gazebo/pause_physics',
        '/gazebo/unpause_physics',
        '/robot1/odom',
        '/robot2/odom',
    ]
    return all(x in text for x in required)


def wait_for_stack_ready(timeout_sec=180):
    start = time.time()
    while time.time() - start < timeout_sec:
        if stack_ready():
            return True
        time.sleep(2)
    return False


def start_stack(log_path: Path):
    log_f = open(log_path, 'w', encoding='utf-8')
    proc = subprocess.Popen(
        ['bash', '-lc', f'cd {PROJECT_ROOT / "scripts"} && bash {MAIN_SCRIPT}'],
        cwd=str(PROJECT_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    return proc, log_f


def stop_stack(proc):
    try:
        _run_bash(f'cd {PROJECT_ROOT / "scripts"} && bash {KILL_SCRIPT} >/dev/null 2>&1 || true')
    except Exception:
        pass
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pattern in [
        'roslaunch sim_env main.launch',
        'gzserver',
        'gzclient',
        'roscore',
        'rosmaster',
    ]:
        _run_bash(f"pkill -f '{pattern}' >/dev/null 2>&1 || true")
    time.sleep(3)


def aggregate_summaries(summaries):
    grouped = {}
    for item in summaries:
        key = (item['model_key'], item['scenario'])
        s = item['summary'] or {}
        grouped.setdefault(key, []).append({
            'repeat_idx': item['repeat_idx'],
            'success_rate': float(s.get('success_rate', 'nan')),
            'collision_rate': float(s.get('collision_rate', 'nan')),
            'avg_reward': float(s.get('avg_reward', 'nan')),
            'timeout_rate': float(s.get('timeout_rate', 'nan')),
        })

    aggregate_rows = []
    for (model_key, scenario), vals in sorted(grouped.items()):
        n = len(vals)
        aggregate_rows.append({
            'model_key': model_key,
            'model_label': MODEL_REGISTRY[model_key]['label'],
            'scenario': scenario,
            'repeats': n,
            'avg_success_rate': sum(v['success_rate'] for v in vals) / n,
            'avg_collision_rate': sum(v['collision_rate'] for v in vals) / n,
            'avg_reward': sum(v['avg_reward'] for v in vals) / n,
            'avg_timeout_rate': sum(v['timeout_rate'] for v in vals) / n,
            'per_repeat': vals,
        })
    return aggregate_rows


def main():
    parser = argparse.ArgumentParser(description='Repeat independent validation for named models with stack restart per job.')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--scenarios', type=str, default='noadv,adv')
    parser.add_argument('--models', type=str, required=True, help='Comma separated keys from MODEL_REGISTRY')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_dir / 'configs'
    csv_dir = out_dir / 'csv'
    stack_log_dir = out_dir / 'stack_logs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    stack_log_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(open(BASE_CONFIG, 'r', encoding='utf-8'))
    adv_cfg_path = cfg_dir / 'forklift_eval_adv.yaml'
    noadv_cfg_path = cfg_dir / 'forklift_eval_noadv.yaml'
    write_yaml(adv_cfg_path, build_eval_config(base_cfg, True, 'movebase'))
    write_yaml(noadv_cfg_path, build_eval_config(base_cfg, False, 'movebase'))

    scenarios = [x.strip() for x in args.scenarios.split(',') if x.strip()]
    model_keys = [x.strip() for x in args.models.split(',') if x.strip()]

    jobs = []
    for model_key in model_keys:
        model_info = MODEL_REGISTRY[model_key]
        for scenario in scenarios:
            for repeat_idx in range(1, args.repeats + 1):
                cfg_path = adv_cfg_path if scenario == 'adv' else noadv_cfg_path
                csv_path = csv_dir / f'{model_key}__{scenario}__repeat{repeat_idx}.csv'
                stack_log = stack_log_dir / f'{model_key}__{scenario}__repeat{repeat_idx}.log'
                cmd = (
                    'source /opt/ros/noetic/setup.bash && '
                    'source /data/lzq/miniconda3/etc/profile.d/conda.sh && '
                    'conda activate rl && '
                    f'source {PROJECT_ROOT / "devel/setup.bash"} && '
                    f'python {EVAL_SCRIPT} '
                    f'--config {cfg_path} '
                    f'--model_path {model_info["model_path"]} '
                    f'--episodes {args.episodes} '
                    f'--csv_path {csv_path}'
                )
                jobs.append({
                    'model_key': model_key,
                    'model_label': model_info['label'],
                    'model_path': str(model_info['model_path']),
                    'scenario': scenario,
                    'repeat_idx': repeat_idx,
                    'csv_path': str(csv_path),
                    'stack_log': str(stack_log),
                    'config_path': str(cfg_path),
                    'command': cmd,
                })

    dump_json(out_dir / 'jobs.json', jobs)

    summaries = []
    summary_json = out_dir / 'repeat_eval_summary.json'
    for i, job in enumerate(jobs, 1):
        csv_path = Path(job['csv_path'])
        if csv_path.exists():
            csv_path.unlink()
        proc = None
        log_f = None
        print(f"[{i}/{len(jobs)}] run {job['model_key']} scenario={job['scenario']} repeat={job['repeat_idx']}", flush=True)
        try:
            proc, log_f = start_stack(Path(job['stack_log']))
            if not wait_for_stack_ready():
                raise RuntimeError(f"stack not ready in time: {job['stack_log']}")
            subprocess.run(['bash', '-lc', job['command']], cwd=str(PROJECT_ROOT), check=True)
            summary = parse_summary(csv_path)
            summaries.append({**job, 'summary': summary})
            dump_json(summary_json, summaries)
        finally:
            if log_f is not None:
                log_f.flush()
            stop_stack(proc)
            if log_f is not None:
                log_f.close()
            time.sleep(2)

    aggregate_rows = aggregate_summaries(summaries)
    dump_json(out_dir / 'repeat_eval_aggregate.json', aggregate_rows)

    with open(out_dir / 'repeat_eval_aggregate.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_key', 'model_label', 'scenario', 'repeats', 'avg_success_rate', 'avg_collision_rate', 'avg_reward', 'avg_timeout_rate'])
        for row in aggregate_rows:
            writer.writerow([
                row['model_key'], row['model_label'], row['scenario'], row['repeats'],
                f"{row['avg_success_rate']:.6f}", f"{row['avg_collision_rate']:.6f}",
                f"{row['avg_reward']:.6f}", f"{row['avg_timeout_rate']:.6f}",
            ])

    print(f'[done] wrote {summary_json}')
    print(f'[done] wrote {out_dir / "repeat_eval_aggregate.json"}')


if __name__ == '__main__':
    main()
