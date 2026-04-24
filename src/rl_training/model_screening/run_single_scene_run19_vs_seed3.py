#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import shutil
import subprocess
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
MODELS = {
    'run19': {
        'label': '课程学习策略',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_19/td3_model',
        'exp_cfg': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_19/experiment_config.yaml',
        'color': '#d62728',
    },
    'seed3': {
        'label': '无课程学习策略',
        'model_path': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_noadv_nocurr_retrain_seed3/td3_model',
        'exp_cfg': PROJECT_ROOT / 'src/rl_training/logs/forklift_movebase/run_noadv_nocurr_retrain_seed3/experiment_config.yaml',
        'color': '#1f77b4',
    },
}


def read_yaml(path: Path):
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
    return subprocess.run(['bash', '-lc', command], cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=check)


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
    required = ['/gazebo/set_model_state', '/gazebo/pause_physics', '/gazebo/unpause_physics', '/robot1/odom']
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
        cwd=str(PROJECT_ROOT), stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid, text=True,
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
    for key in ['collision_dist', 'control_dt', 'goal_reached_dist', 'num_actions', 'num_observations', 'map']:
        if key in trained_env:
            env[key] = trained_env[key]
    env['debug'] = False
    env['max_episode_length'] = int(max_episode_length)
    env['collision_dist'] = 0.4
    env['goal_reached_dist'] = 0.3
    env['opponent']['enabled'] = False
    env['goal_mode'] = 'random'
    env['goal_range'] = {'x_min': -6.5, 'x_max': -1.0, 'y_min': -5.0, 'y_max': -1.0}
    env['curriculum'] = {'delta': 0.0, 'initial_span': 5.0, 'max_span': 5.0}
    env['ego_spawn_radius'] = [1.0, 2.0]
    return cfg


def run_eval(command: str, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w', encoding='utf-8') as f:
        proc = subprocess.run(['bash', '-lc', command], cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'eval command failed: {log_path}')


def load_traj(traj_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    ego = []
    adv = []
    with open(traj_csv, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ego.append((float(row['ego_x']), float(row['ego_y'])))
            adv.append((float(row['adv_x']), float(row['adv_y'])))
    return np.array(ego, dtype=float), np.array(adv, dtype=float)


def effective_progress_rate(points: np.ndarray, goal: np.ndarray, eps: float = 0.01) -> float:
    if len(points) < 2:
        return 0.0
    d = np.linalg.norm(points - goal[None, :], axis=1)
    progress = (d[:-1] - d[1:]) > eps
    return float(progress.mean()) if len(progress) else 0.0


def stagnation_ratio(points: np.ndarray, move_thresh: float = 0.03) -> float:
    if len(points) < 2:
        return 1.0
    step_move = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float((step_move < move_thresh).mean())


def direction_oscillation_rate(points: np.ndarray, move_thresh: float = 0.03, turn_thresh_deg: float = 8.0) -> float:
    if len(points) < 3:
        return 0.0
    vecs = np.diff(points, axis=0)
    valid = np.linalg.norm(vecs, axis=1) > move_thresh
    vecs = vecs[valid]
    if len(vecs) < 2:
        return 0.0
    signs = []
    for i in range(len(vecs) - 1):
        v1 = vecs[i]
        v2 = vecs[i + 1]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = float(np.dot(v1, v2))
        ang = math.degrees(math.atan2(abs(cross), max(dot, 1e-9)))
        if ang < turn_thresh_deg:
            continue
        signs.append(1 if cross > 0 else -1)
    if len(signs) < 2:
        return 0.0
    changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    return float(changes / (len(signs) - 1))


def first_stable_progress_step(points: np.ndarray, goal: np.ndarray, window: int = 5, eps: float = 0.01) -> int:
    if len(points) < window + 1:
        return len(points)
    d = np.linalg.norm(points - goal[None, :], axis=1)
    progress = (d[:-1] - d[1:]) > eps
    for i in range(0, len(progress) - window + 1):
        if progress[i:i + window].all():
            return i + 1
    return len(points)


def compute_metrics(points: np.ndarray, goal_xy: Tuple[float, float]) -> Dict[str, float]:
    goal = np.array(goal_xy, dtype=float)
    return {
        'effective_progress_rate': effective_progress_rate(points, goal),
        'stagnation_ratio': stagnation_ratio(points),
        'direction_oscillation_rate': direction_oscillation_rate(points),
        'first_stable_progress_step': float(first_stable_progress_step(points, goal)),
    }


def setup_fonts():
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def load_local_obstacle_points(map_yaml_path: Path, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> np.ndarray:
    meta = read_yaml(map_yaml_path)
    img_path = map_yaml_path.parent / meta['image']
    img = np.array(Image.open(img_path).convert('L'))
    res = float(meta['resolution'])
    origin_x, origin_y = float(meta['origin'][0]), float(meta['origin'][1])
    occ_thresh = float(meta.get('occupied_thresh', 0.65))
    negate = int(meta.get('negate', 0))

    if negate == 0:
        occ = (255.0 - img.astype(np.float32)) / 255.0
    else:
        occ = img.astype(np.float32) / 255.0
    occ_mask = occ >= occ_thresh
    ys, xs = np.where(occ_mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=float)

    world_x = origin_x + (xs + 0.5) * res
    h = img.shape[0]
    world_y = origin_y + (h - ys - 0.5) * res

    keep = (
        (world_x >= xlim[0]) & (world_x <= xlim[1]) &
        (world_y >= ylim[0]) & (world_y <= ylim[1])
    )
    if not np.any(keep):
        return np.zeros((0, 2), dtype=float)
    return np.stack([world_x[keep], world_y[keep]], axis=1)


def plot_overlay(scene: Dict, trajectories: Dict[str, List[np.ndarray]], out_png: Path):
    setup_fonts()
    fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=240)
    start = (scene['ego_x'], scene['ego_y'])
    goal = (scene['goal_x'], scene['goal_y'])

    all_pts = [np.array([[start[0], start[1]], [goal[0], goal[1]]], dtype=float)]
    for traj_list in trajectories.values():
        all_pts.extend([pts for pts in traj_list if len(pts) > 0])
    all_pts = np.concatenate(all_pts, axis=0)
    margin = 0.35
    xlim = (float(np.min(all_pts[:, 0]) - margin), float(np.max(all_pts[:, 0]) + margin))
    ylim = (float(np.min(all_pts[:, 1]) - margin), float(np.max(all_pts[:, 1]) + margin))
    obstacle_pts = load_local_obstacle_points(MAP_YAML, xlim, ylim)
    if len(obstacle_pts) > 0:
        ax.scatter(
            obstacle_pts[:, 0], obstacle_pts[:, 1],
            s=5, c='black', marker='s', alpha=0.32, linewidths=0, label='障碍物', zorder=1
        )

    ax.scatter([start[0]], [start[1]], c='#2ca02c', marker='s', s=52, label='起点', zorder=5)
    ax.scatter([goal[0]], [goal[1]], c='#9467bd', marker='*', s=120, label='目标点', zorder=5)

    for key, traj_list in trajectories.items():
        info = MODELS[key]
        for idx, pts in enumerate(traj_list):
            if len(pts) == 0:
                continue
            xs, ys = pts[:, 0], pts[:, 1]
            ax.plot(
                xs, ys,
                color=info['color'],
                lw=2.0,
                alpha=0.55 if len(traj_list) > 1 else 0.95,
                label=info['label'] if idx == 0 else None,
                zorder=3,
            )
            ax.scatter([xs[-1]], [ys[-1]], c=info['color'], s=18, marker='o', alpha=0.75, zorder=4)

    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle='--', alpha=0.22)
    ax.legend(loc='best', fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def plot_metrics(metrics_rows: List[Dict], out_png: Path):
    setup_fonts()
    labels = ['有效推进率', '停滞步数占比', '方向振荡率', '首次稳定推进步数']
    m1 = metrics_rows[0]
    m2 = metrics_rows[1]
    vals1 = [m1['effective_progress_rate'], m1['stagnation_ratio'], m1['direction_oscillation_rate'], m1['first_stable_progress_step']]
    vals2 = [m2['effective_progress_rate'], m2['stagnation_ratio'], m2['direction_oscillation_rate'], m2['first_stable_progress_step']]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=220)
    ax.bar(x - width / 2, vals1, width, color=MODELS[m1['model_key']]['color'], label=MODELS[m1['model_key']]['label'])
    ax.bar(x + width / 2, vals2, width, color=MODELS[m2['model_key']]['color'], label=MODELS[m2['model_key']]['label'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)
    for i, v in enumerate(vals1):
        ax.text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(vals2):
        ax.text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Single-scene comparison: run_19 vs seed3')
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'src/rl_training/model_screening/single_scene_run19_vs_seed3_20260406'))
    parser.add_argument('--goal-x', type=float, default=-5.6037463375963315)
    parser.add_argument('--goal-y', type=float, default=-3.1052022614959838)
    parser.add_argument('--ego-x', type=float, default=-6.990766497438723)
    parser.add_argument('--ego-y', type=float, default=-5.289383250911921)
    parser.add_argument('--ego-yaw', type=float, default=1.00)
    parser.add_argument('--max-steps', type=int, default=180)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()

    scene = {
        'goal_x': args.goal_x,
        'goal_y': args.goal_y,
        'ego_x': args.ego_x,
        'ego_y': args.ego_y,
        'ego_yaw': args.ego_yaw,
        'goal_yaw': math.atan2(args.goal_y - args.ego_y, args.goal_x - args.ego_x),
    }

    out_dir = Path(args.output_dir)
    cfg_dir = out_dir / 'configs'
    log_dir = out_dir / 'logs'
    csv_dir = out_dir / 'csv'
    traj_dir = out_dir / 'trajectories'
    fig_dir = out_dir / 'figures'
    for d in [cfg_dir, log_dir, csv_dir, traj_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    setup_json = out_dir / 'fixed_scene.json'
    dump_json(setup_json, [scene for _ in range(args.repeats)])

    base_cfg = read_yaml(BASE_CONFIG)
    metrics_rows = []
    trajectories: Dict[str, List[np.ndarray]] = {}

    for idx, key in enumerate(['run19', 'seed3'], start=1):
        info = MODELS[key]
        exp_cfg = read_yaml(info['exp_cfg'])
        cfg = merged_eval_config(base_cfg, exp_cfg, args.max_steps)
        cfg_path = cfg_dir / f'{key}.yaml'
        write_yaml(cfg_path, cfg)
        csv_path = csv_dir / f'{key}.csv'
        traj_root = traj_dir / key
        if traj_root.exists():
            shutil.rmtree(traj_root)
        cmd = (
            'source /opt/ros/noetic/setup.bash && '
            'source /data/lzq/miniconda3/etc/profile.d/conda.sh && '
            'conda activate rl && '
            f'source {PROJECT_ROOT / "devel/setup.bash"} && '
            f'python {EVAL_SCRIPT} --config {cfg_path} --model_path {info["model_path"]} '
            f'--episodes {args.repeats} --random_seed 20260406 --episode_setup_json {setup_json} '
            f'--csv_path {csv_path} --record_trajectory_output_dir {traj_root} --record_all_trajectory_episodes'
        )
        proc = None
        log_f = None
        try:
            proc, log_f = start_stack(log_dir / f'{idx:02d}_{key}_stack.log')
            if not wait_for_stack_ready():
                raise RuntimeError(f'stack not ready for {key}')
            run_eval(cmd, log_dir / f'{idx:02d}_{key}_eval.log')
        finally:
            if log_f is not None:
                log_f.flush(); log_f.close()
            stop_stack(proc)

        trajectories[key] = []
        episode_dirs = sorted([p for p in traj_root.glob('episode_*') if p.is_dir()])
        for ep_idx, ep_dir in enumerate(episode_dirs, start=1):
            traj_csv = ep_dir / 'trajectory.csv'
            meta_json = ep_dir / 'trajectory_meta.json'
            if not traj_csv.exists() or not meta_json.exists():
                continue
            ego_pts, _ = load_traj(traj_csv)
            trajectories[key].append(ego_pts)
            meta = json.loads(meta_json.read_text(encoding='utf-8'))
            metrics = compute_metrics(ego_pts, (scene['goal_x'], scene['goal_y']))
            metrics.update({
                'model_key': key,
                'model_label': info['label'],
                'repeat_id': ep_idx,
                'collided': int(bool(meta.get('collided', False))),
                'reached_goal': int(bool(meta.get('reached_goal', False))),
                'steps': int(meta.get('steps', 0)),
                'episode_reward': float(meta.get('episode_reward', 0.0)),
                'final_goal_distance': float(np.linalg.norm(ego_pts[-1] - np.array([scene['goal_x'], scene['goal_y']]))) if len(ego_pts) else float('nan'),
                'traj_csv': str(traj_csv),
                'meta_json': str(meta_json),
            })
            metrics_rows.append(metrics)

    plot_overlay(scene, trajectories, fig_dir / 'run19_vs_seed3_single_scene_overlay.png')
    if len(metrics_rows) >= 2:
        summary_rows = []
        for key in ['run19', 'seed3']:
            rows = [r for r in metrics_rows if r['model_key'] == key]
            if not rows:
                continue
            agg = {
                'model_key': key,
                'model_label': MODELS[key]['label'],
                'effective_progress_rate': float(np.mean([r['effective_progress_rate'] for r in rows])),
                'stagnation_ratio': float(np.mean([r['stagnation_ratio'] for r in rows])),
                'direction_oscillation_rate': float(np.mean([r['direction_oscillation_rate'] for r in rows])),
                'first_stable_progress_step': float(np.mean([r['first_stable_progress_step'] for r in rows])),
            }
            summary_rows.append(agg)
        if len(summary_rows) == 2:
            plot_metrics(summary_rows, fig_dir / 'run19_vs_seed3_behavior_metrics.png')
    write_csv(out_dir / 'behavior_metrics.csv', list(metrics_rows[0].keys()), metrics_rows)
    dump_json(out_dir / 'summary.json', {
        'scene': scene,
        'repeats': args.repeats,
        'metrics': metrics_rows,
        'overlay_figure': str(fig_dir / 'run19_vs_seed3_single_scene_overlay.png'),
        'metrics_figure': str(fig_dir / 'run19_vs_seed3_behavior_metrics.png'),
    })
    print(json.dumps({'scene': scene, 'metrics': metrics_rows}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
