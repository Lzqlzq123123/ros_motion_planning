#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path('/data/lzq/ros_motion_planning')
BASE_CONFIG = PROJECT_ROOT / 'src/rl_training/config/forklift_movebase.yaml'
EVAL_SCRIPT = PROJECT_ROOT / 'src/rl_training/eval_velodyne_td3_with_goal.py'
MAIN_SCRIPT = PROJECT_ROOT / 'scripts/main.sh'
KILL_SCRIPT = PROJECT_ROOT / 'scripts/killpro.sh'
MAP_YAML = PROJECT_ROOT / 'src/sim_env/maps/warehouse/warehouse.yaml'

MODEL_REGISTRY = {
    'nocurr_seed1': {
        'label': '无课程学习',
        'group': 'nocurr',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_noadv_nocurr_seed1/td3_model',
        'exp_cfg': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_noadv_nocurr_seed1/experiment_config.yaml',
    },
    'nocurr_seed2': {
        'label': '无课程学习',
        'group': 'nocurr',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_noadv_nocurr_seed2/td3_model',
        'exp_cfg': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_noadv_nocurr_seed2/experiment_config.yaml',
    },
    'nocurr_seed3': {
        'label': '无课程学习',
        'group': 'nocurr',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_noadv_nocurr_retrain_seed3/td3_model',
        'exp_cfg': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_noadv_nocurr_retrain_seed3/experiment_config.yaml',
    },
    'curr_run19': {
        'label': '课程学习',
        'group': 'curr',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_19/td3_model',
        'exp_cfg': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_19/experiment_config.yaml',
    },
    'curr_back_run7': {
        'label': '课程学习',
        'group': 'curr',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/back_run_7/td3_model',
        'exp_cfg': None,
    },
}


def read_yaml(path: Path):
    if path is None or not Path(path).exists():
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def dump_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
        'source /opt/ros/noetic/setup.bash && '
        'rosservice list 2>/dev/null && '
        "echo '---TOPICS---' && "
        'rostopic list 2>/dev/null'
    )
    if probe.returncode != 0:
        return False
    text = probe.stdout
    required = [
        '/gazebo/set_model_state', '/gazebo/pause_physics', '/gazebo/unpause_physics',
        '/robot1/odom',
    ]
    return all(x in text for x in required)


def wait_for_stack_ready(timeout_sec=240):
    start = time.time()
    while time.time() - start < timeout_sec:
        if stack_ready():
            return True
        time.sleep(2)
    return False


def start_stack(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, 'w', encoding='utf-8')
    proc = subprocess.Popen(
        ['bash', '-lc', f'source /opt/ros/noetic/setup.bash && cd {PROJECT_ROOT / "scripts"} && bash {MAIN_SCRIPT}'],
        cwd=str(PROJECT_ROOT), stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid, text=True,
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
    time.sleep(2)
    for pattern in ['roslaunch sim_env main.launch', 'gzserver', 'gzclient', 'roscore', 'rosmaster']:
        _run_bash(f"pkill -f '{pattern}' >/dev/null 2>&1 || true")
    time.sleep(2)


def merged_eval_config(base_cfg: dict, exp_cfg: dict, max_episode_length: int) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    env = cfg['env']
    trained_env = exp_cfg.get('environment', {})
    for key in [
        'collision_dist', 'control_dt', 'goal_reached_dist', 'num_actions',
        'num_observations', 'map'
    ]:
        if key in trained_env:
            env[key] = trained_env[key]

    env['debug'] = False
    env['max_episode_length'] = int(max_episode_length)
    # Use unified evaluation thresholds across all compared models for fair comparison.
    env['collision_dist'] = 0.4
    env['goal_reached_dist'] = 0.3
    env['opponent']['enabled'] = False
    # fixed setups decide actual goal/start; still use full-difficulty range for completeness
    env['goal_mode'] = 'random'
    env['goal_range'] = {
        'x_min': -6.5,
        'x_max': -1.0,
        'y_min': -5.0,
        'y_max': -1.0,
    }
    env['curriculum'] = {'delta': 0.0, 'initial_span': 5.0, 'max_span': 5.0}
    env['ego_spawn_radius'] = [2.5, 4.0]
    return cfg


def generator_config(base_cfg: dict, setup_max_episode_length: int) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    env = cfg['env']
    env['debug'] = False
    env['max_episode_length'] = int(setup_max_episode_length)
    env['opponent']['enabled'] = False
    env['goal_mode'] = 'random'
    env['goal_range'] = {
        'x_min': -6.5,
        'x_max': -1.0,
        'y_min': -5.0,
        'y_max': -1.0,
    }
    env['curriculum'] = {'delta': 0.0, 'initial_span': 5.0, 'max_span': 5.0}
    env['ego_spawn_radius'] = [2.5, 4.0]
    return cfg


def run_eval_command(command: str, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        proc = subprocess.run(['bash', '-lc', command], cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'eval command failed, see {log_path}')


def load_trajectory(traj_csv: Path) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if not traj_csv.exists():
        return pts
    with open(traj_csv, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row['ego_x']), float(row['ego_y'])))
    return pts


def path_length(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i-1][0], points[i][1] - points[i-1][1])
    return total


def parse_episode_metrics(model_key: str, traj_root: Path, setups: List[Dict]) -> List[Dict]:
    rows = []
    for idx, setup in enumerate(setups, start=1):
        ep_dir = traj_root / f'episode_{idx:03d}'
        meta_path = ep_dir / 'trajectory_meta.json'
        traj_csv = ep_dir / 'trajectory.csv'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        pts = load_trajectory(traj_csv)
        goal_x = float(meta.get('goal_x', setup['goal_x']))
        goal_y = float(meta.get('goal_y', setup['goal_y']))
        start_x = float(meta.get('ego_x', setup['ego_x']))
        start_y = float(meta.get('ego_y', setup['ego_y']))
        final_x, final_y = pts[-1] if pts else (start_x, start_y)
        final_goal_dist = math.hypot(final_x - goal_x, final_y - goal_y)
        direct_dist = math.hypot(start_x - goal_x, start_y - goal_y)
        actual_len = path_length(pts)
        efficiency = (direct_dist / actual_len) if actual_len > 1e-6 else 0.0
        rows.append({
            'model_key': model_key,
            'episode': idx,
            'collided': int(bool(meta.get('collided', False))),
            'reached_goal': int(bool(meta.get('reached_goal', False))),
            'timed_out': int(bool(meta.get('timed_out', False))),
            'steps': int(meta.get('steps', 0)),
            'episode_reward': float(meta.get('episode_reward', 0.0)),
            'goal_x': goal_x,
            'goal_y': goal_y,
            'ego_x': start_x,
            'ego_y': start_y,
            'final_x': final_x,
            'final_y': final_y,
            'direct_distance': direct_dist,
            'path_length': actual_len,
            'path_efficiency': efficiency,
            'final_goal_distance': final_goal_dist,
            'traj_csv': str(traj_csv),
            'meta_json': str(meta_path),
        })
    return rows


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float('nan')


def safe_std(values: List[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def aggregate_metrics(rows: List[Dict], model_key: str) -> Dict:
    success = [r['reached_goal'] for r in rows]
    collision = [r['collided'] for r in rows]
    steps_success = [r['steps'] for r in rows if r['reached_goal']]
    path_eff = [r['path_efficiency'] for r in rows]
    final_dist = [r['final_goal_distance'] for r in rows]
    rewards = [r['episode_reward'] for r in rows]
    return {
        'model_key': model_key,
        'episodes': len(rows),
        'success_rate': safe_mean(success),
        'collision_rate': safe_mean(collision),
        'avg_steps_on_success': safe_mean(steps_success),
        'avg_path_efficiency': safe_mean(path_eff),
        'std_path_efficiency': safe_std(path_eff),
        'avg_final_goal_distance': safe_mean(final_dist),
        'std_final_goal_distance': safe_std(final_dist),
        'avg_episode_reward': safe_mean(rewards),
    }


def rank_summary(summary: Dict) -> Tuple:
    return (
        summary['success_rate'],
        -summary['collision_rate'],
        -summary['avg_final_goal_distance'],
        summary['avg_path_efficiency'],
        -summary['std_final_goal_distance'],
    )


def choose_best_models(summary_rows: List[Dict]) -> Dict[str, Dict]:
    best = {}
    for group in ['curr', 'nocurr']:
        candidates = [r for r in summary_rows if MODEL_REGISTRY[r['model_key']]['group'] == group]
        best[group] = max(candidates, key=rank_summary)
    return best


def choose_representative_episode(curr_rows: List[Dict], nocurr_rows: List[Dict]) -> int:
    curr_by_ep = {r['episode']: r for r in curr_rows}
    nocurr_by_ep = {r['episode']: r for r in nocurr_rows}
    candidate = None
    candidate_score = None
    for ep in sorted(set(curr_by_ep) & set(nocurr_by_ep)):
        c = curr_by_ep[ep]
        n = nocurr_by_ep[ep]
        score = (
            100 * (c['reached_goal'] - n['reached_goal'])
            + 40 * (n['collided'] - c['collided'])
            + (n['final_goal_distance'] - c['final_goal_distance'])
            - 0.01 * abs(c['steps'] - n['steps'])
        )
        if candidate_score is None or score > candidate_score:
            candidate_score = score
            candidate = ep
    return int(candidate or 1)


def load_map_image_and_extent(map_yaml_path: Path):
    meta = read_yaml(map_yaml_path)
    image_path = map_yaml_path.parent / meta['image']
    img = Image.open(image_path)
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    res = float(meta['resolution'])
    origin_x, origin_y = float(meta['origin'][0]), float(meta['origin'][1])
    h, w = arr.shape[:2]
    extent = [origin_x, origin_x + w * res, origin_y, origin_y + h * res,]
    return arr, extent


def setup_matplotlib_fonts():
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def plot_metric_compare(best_curr: Dict, best_nocurr: Dict, out_png: Path):
    setup_matplotlib_fonts()
    labels = ['成功率', '碰撞率', '平均路径效率', '平均目标误差']
    curr_values = [
        100 * best_curr['success_rate'],
        100 * best_curr['collision_rate'],
        best_curr['avg_path_efficiency'],
        best_curr['avg_final_goal_distance'],
    ]
    nocurr_values = [
        100 * best_nocurr['success_rate'],
        100 * best_nocurr['collision_rate'],
        best_nocurr['avg_path_efficiency'],
        best_nocurr['avg_final_goal_distance'],
    ]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=220)
    ax.bar(x - width/2, nocurr_values, width, label='无课程学习', color='#4e79a7')
    ax.bar(x + width/2, curr_values, width, label='课程学习', color='#e15759')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    for i, v in enumerate(nocurr_values):
        ax.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(curr_values):
        ax.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def _plot_single_scene(ax, map_arr, extent, points, row, legend_label):
    ax.imshow(map_arr, origin='lower', extent=extent, alpha=0.92)
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color='#d62728' if row['reached_goal'] else '#4e79a7', lw=2.4, label=legend_label)
        ax.scatter(xs[0], ys[0], c='#2ca02c', marker='s', s=40, label='起点')
        ax.scatter(row['goal_x'], row['goal_y'], c='#9467bd', marker='*', s=90, label='目标点')
        ax.scatter(xs[-1], ys[-1], c='#ff7f0e', marker='o', s=34, label='终点')
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')


def plot_scene_compare(curr_row: Dict, nocurr_row: Dict, out_png: Path):
    setup_matplotlib_fonts()
    map_arr, extent = load_map_image_and_extent(MAP_YAML)
    curr_pts = load_trajectory(Path(curr_row['traj_csv']))
    nocurr_pts = load_trajectory(Path(nocurr_row['traj_csv']))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3), dpi=220)
    _plot_single_scene(axes[0], map_arr, extent, nocurr_pts, nocurr_row, '无课程学习轨迹')
    _plot_single_scene(axes[1], map_arr, extent, curr_pts, curr_row, '课程学习轨迹')
    axes[0].legend(loc='lower left', fontsize=9)
    axes[1].legend(loc='lower left', fontsize=9)
    axes[0].text(0.02, 0.98, f"是否到达: {'是' if nocurr_row['reached_goal'] else '否'}\n碰撞: {'是' if nocurr_row['collided'] else '否'}\n最终目标误差: {nocurr_row['final_goal_distance']:.2f} m", transform=axes[0].transAxes, va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
    axes[1].text(0.02, 0.98, f"是否到达: {'是' if curr_row['reached_goal'] else '否'}\n碰撞: {'是' if curr_row['collided'] else '否'}\n最终目标误差: {curr_row['final_goal_distance']:.2f} m", transform=axes[1].transAxes, va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Pure-ego curriculum vs no-curriculum fixed-scene comparison')
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'src/rl_training/model_screening/curriculum_noadv_compare_20260405'))
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--max-steps', type=int, default=180)
    parser.add_argument('--generator-max-episode-length', type=int, default=1)
    parser.add_argument('--fixed-setups-json', type=str, default=None)
    parser.add_argument('--models', type=str, default=None, help='Comma-separated model keys to run; default runs all registry entries')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    cfg_dir = out_dir / 'configs'
    log_dir = out_dir / 'logs'
    csv_dir = out_dir / 'csv'
    traj_dir = out_dir / 'trajectories'
    fig_dir = out_dir / 'figures'
    for d in [cfg_dir, log_dir, csv_dir, traj_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    base_cfg = read_yaml(BASE_CONFIG)

    if args.fixed_setups_json:
        fixed_setup_json = Path(args.fixed_setups_json)
        setups = json.loads(fixed_setup_json.read_text(encoding='utf-8'))
        dump_json(out_dir / 'setup_generation_summary.json', {
            'episodes': len(setups),
            'setup_json': str(fixed_setup_json),
            'generator_config': None,
            'reused': True,
        })
    else:
        # 1) generate shared fixed episode setups under full-difficulty no-adv configuration
        fixed_setup_json = out_dir / 'fixed_episode_setups.json'
        generator_cfg_path = cfg_dir / 'generator_noadv.yaml'
        write_yaml(generator_cfg_path, generator_config(base_cfg, args.generator_max_episode_length))
        generator_csv = csv_dir / 'setup_generator.csv'
        generator_cmd = (
            'source /opt/ros/noetic/setup.bash && '
            'source /data/lzq/miniconda3/etc/profile.d/conda.sh && '
            'conda activate rl && '
            f'source {PROJECT_ROOT / "devel/setup.bash"} && '
            f'python {EVAL_SCRIPT} --config {generator_cfg_path} --robot1_mode movebase '
            f'--episodes {args.episodes} --random_seed 20260405 '
            f'--csv_path {generator_csv} --save_episode_setups_json {fixed_setup_json}'
        )

        proc = None
        log_f = None
        try:
            proc, log_f = start_stack(log_dir / '00_setup_generator_stack.log')
            if not wait_for_stack_ready():
                raise RuntimeError('ROS/Gazebo stack did not become ready while generating shared setups')
            run_eval_command(generator_cmd, log_dir / '00_setup_generator_eval.log')
        finally:
            if log_f is not None:
                log_f.flush(); log_f.close()
            stop_stack(proc)

        setups = json.loads(fixed_setup_json.read_text(encoding='utf-8'))
        dump_json(out_dir / 'setup_generation_summary.json', {
            'episodes': len(setups),
            'setup_json': str(fixed_setup_json),
            'generator_config': str(generator_cfg_path),
            'reused': False,
        })

    all_episode_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    selected_model_keys = [k.strip() for k in args.models.split(',')] if args.models else list(MODEL_REGISTRY.keys())

    # 2) evaluate four candidate models on the same fixed setups
    for idx, model_key in enumerate(selected_model_keys, start=1):
        info = MODEL_REGISTRY[model_key]
        exp_cfg = read_yaml(info['exp_cfg'])
        eval_cfg = merged_eval_config(base_cfg, exp_cfg, args.max_steps)
        cfg_path = cfg_dir / f'{model_key}.yaml'
        write_yaml(cfg_path, eval_cfg)
        model_csv = csv_dir / f'{model_key}.csv'
        model_traj_dir = traj_dir / model_key
        cmd = (
            'source /opt/ros/noetic/setup.bash && '
            'source /data/lzq/miniconda3/etc/profile.d/conda.sh && '
            'conda activate rl && '
            f'source {PROJECT_ROOT / "devel/setup.bash"} && '
            f'python {EVAL_SCRIPT} --config {cfg_path} --model_path {info["model_path"]} '
            f'--episodes {args.episodes} --random_seed 20260405 '
            f'--episode_setup_json {fixed_setup_json} --csv_path {model_csv} '
            f'--record_trajectory_output_dir {model_traj_dir} --record_all_trajectory_episodes'
        )
        proc = None
        log_f = None
        try:
            proc, log_f = start_stack(log_dir / f'{idx:02d}_{model_key}_stack.log')
            if not wait_for_stack_ready():
                raise RuntimeError(f'ROS/Gazebo stack did not become ready for {model_key}')
            run_eval_command(cmd, log_dir / f'{idx:02d}_{model_key}_eval.log')
        finally:
            if log_f is not None:
                log_f.flush(); log_f.close()
            stop_stack(proc)

        ep_rows = parse_episode_metrics(model_key, model_traj_dir, setups)
        for row in ep_rows:
            row['model_label'] = info['label']
            row['group'] = info['group']
        all_episode_rows.extend(ep_rows)
        agg = aggregate_metrics(ep_rows, model_key)
        agg['model_label'] = info['label']
        agg['group'] = info['group']
        agg['model_path'] = str(info['model_path'])
        summary_rows.append(agg)

    write_csv(out_dir / 'per_episode_metrics.csv', list(all_episode_rows[0].keys()), all_episode_rows)
    write_csv(out_dir / 'aggregate_metrics.csv', list(summary_rows[0].keys()), summary_rows)

    best = choose_best_models(summary_rows)
    best_curr = best['curr']
    best_nocurr = best['nocurr']

    best_curr_rows = [r for r in all_episode_rows if r['model_key'] == best_curr['model_key']]
    best_nocurr_rows = [r for r in all_episode_rows if r['model_key'] == best_nocurr['model_key']]
    rep_episode = choose_representative_episode(best_curr_rows, best_nocurr_rows)
    curr_rep = next(r for r in best_curr_rows if r['episode'] == rep_episode)
    nocurr_rep = next(r for r in best_nocurr_rows if r['episode'] == rep_episode)

    plot_metric_compare(best_curr, best_nocurr, fig_dir / 'curriculum_noadv_metric_compare.png')
    plot_scene_compare(curr_rep, nocurr_rep, fig_dir / 'curriculum_noadv_scene_compare.png')

    final_report = {
        'best_curriculum_model': best_curr,
        'best_no_curriculum_model': best_nocurr,
        'representative_episode': rep_episode,
        'representative_curriculum_episode': curr_rep,
        'representative_no_curriculum_episode': nocurr_rep,
        'figure_metric_compare': str(fig_dir / 'curriculum_noadv_metric_compare.png'),
        'figure_scene_compare': str(fig_dir / 'curriculum_noadv_scene_compare.png'),
    }
    dump_json(out_dir / 'final_report.json', final_report)
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
