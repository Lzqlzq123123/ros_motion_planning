#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import yaml

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl_training.model_screening.run_table31_movebase_eval import (
    BASE_CONFIG,
    EVAL_SCRIPT,
    USER_CONFIG,
    NOMAD_SERVICE_PARAMS,
    load_yaml,
    write_yaml,
    deep_set,
    set_robot_planners,
    start_stack,
    stop_stack,
    wait_for_stack_ready,
    start_nomad_service,
    stop_nomad_service,
    wait_for_ros_service,
)


OUT_DEFAULT = PROJECT_ROOT / "src/rl_training/model_screening/fixed_goal_m3_m3_global_plan_overlay_20260324"
UNCOND_WEIGHTS = "/data/lzq/visualnav-transformer/train/logs/diffusion_2d_ablation/diffusion_2d_gazebo_20260310_212252/latest.pth"
UNCOND_CONFIG = "/data/lzq/visualnav-transformer/train/config/diffusion_2d.yaml"


def run_bash_live(command: str, log_path: Path, cwd: Path = PROJECT_ROOT) -> int:
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(["bash", "-lc", command], cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
        return proc.wait()


def start_capture(topic: str, out_yaml: Path, timeout_sec: int = 90):
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"source /opt/ros/noetic/setup.bash && "
        f"timeout {timeout_sec}s rostopic echo -n 1 {topic} > {out_yaml}"
    )
    return subprocess.Popen(["bash", "-lc", cmd], cwd=str(PROJECT_ROOT), preexec_fn=os.setsid)


def stop_proc(proc):
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def parse_path_yaml(path: Path) -> List[Tuple[float, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    docs = [doc for doc in yaml.safe_load_all(text) if isinstance(doc, dict)]
    if not docs:
        return []
    data = docs[-1]
    poses = data.get("poses", []) if isinstance(data, dict) else []
    pts = []
    for item in poses:
        try:
            pos = item["pose"]["position"]
            pts.append((float(pos["x"]), float(pos["y"])))
        except Exception:
            continue
    return pts


def read_eval_traj_adv(csv_path: Path) -> List[Tuple[float, float]]:
    pts = []
    if not csv_path.exists():
        return pts
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pts.append((float(row["adv_x"]), float(row["adv_y"])))
    return pts


def save_points_csv(path: Path, points: List[Tuple[float, float]], method: str, agent: str, source_type: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "agent", "source_type", "point_idx", "x", "y"])
        for i, (x, y) in enumerate(points):
            writer.writerow([method, agent, source_type, i, x, y])


def append_combined(combined_rows: List[List], points: List[Tuple[float, float]], method: str, agent: str, source_type: str):
    for i, (x, y) in enumerate(points):
        combined_rows.append([method, agent, source_type, i, x, y])


def make_nomad_params(base_path: Path, overrides: Dict, out_path: Path) -> Path:
    cfg = load_yaml(base_path)
    for k, v in overrides.items():
        deep_set(cfg, k, v)
    write_yaml(out_path, cfg)
    return out_path


def plot_overlay(all_paths: Dict, out_png: Path, goal_xy: Tuple[float, float]):
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(7.2, 6.2), dpi=200)
    colors = {
        "ego_movebase": "#222222",
        "movebase": "#1f77b4",
        "diffusion": "#d62728",
        "unconditional_diffusion": "#ff7f0e",
        "rule_based": "#2ca02c",
    }
    labels = {
        "ego_movebase": "自车 movebase 全局轨迹",
        "movebase": "对抗车 movebase 全局轨迹",
        "diffusion": "对抗车扩散模型全局轨迹",
        "unconditional_diffusion": "对抗车无条件扩散全局轨迹",
        "rule_based": "对抗车规则方法执行轨迹",
    }

    order = ["ego_movebase", "movebase", "diffusion", "unconditional_diffusion", "rule_based"]
    for key in order:
        pts = all_paths.get(key, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", markersize=2.2, linewidth=2.0, color=colors[key], label=labels[key])
        plt.scatter(xs[0], ys[0], color=colors[key], s=28, marker="s")
        plt.scatter(xs[-1], ys[-1], color=colors[key], s=36, marker="*")

    plt.scatter([goal_xy[0]], [goal_xy[1]], c="#9467bd", s=65, marker="X", label=f"共享目标点 ({goal_xy[0]:g}, {goal_xy[1]:g})")
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run fixed-goal one-episode overlay for four adversary methods.")
    parser.add_argument("--output-dir", type=str, default=str(OUT_DEFAULT))
    parser.add_argument("--goal-x", type=float, default=-3.0)
    parser.add_argument("--goal-y", type=float, default=-3.0)
    parser.add_argument("--ego-x", type=float, default=-5.0)
    parser.add_argument("--ego-y", type=float, default=-3.0)
    parser.add_argument("--ego-yaw", type=float, default=0.0)
    parser.add_argument("--adv-x", type=float, default=-2.7)
    parser.add_argument("--adv-y", type=float, default=-1.5)
    parser.add_argument("--adv-yaw", type=float, default=-1.75)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--robot2-classical-global-planner", type=str, default="theta_star")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "configs").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "captures").mkdir(exist_ok=True)
    (out_dir / "paths").mkdir(exist_ok=True)
    (out_dir / "csv").mkdir(exist_ok=True)
    (out_dir / "trajectories").mkdir(exist_ok=True)

    fixed_setup = [{
        "goal_x": args.goal_x,
        "goal_y": args.goal_y,
        "goal_yaw": 0.0,
        "ego_x": args.ego_x,
        "ego_y": args.ego_y,
        "ego_yaw": args.ego_yaw,
        "adv_x": args.adv_x,
        "adv_y": args.adv_y,
        "adv_yaw": args.adv_yaw,
    }]
    setup_json = out_dir / "fixed_episode_setup.json"
    setup_json.write_text(json.dumps(fixed_setup, ensure_ascii=False, indent=2), encoding="utf-8")

    base_cfg = load_yaml(BASE_CONFIG)
    methods = [
        {
            "key": "movebase",
            "opponent_mode": "movebase",
            "robot2_global": args.robot2_classical_global_planner,
            "robot2_local": "pid",
            "goal_offset": 0.0,
            "nomad_overrides": None,
            "adv_source": "plan",
        },
        {
            "key": "diffusion",
            "opponent_mode": "diffusion",
            "robot2_global": "diffusion",
            "robot2_local": "pid",
            "goal_offset": 0.0,
            "nomad_overrides": {
                "classical_guidance_weight": 0.15,
                "opponent_guidance_enabled": True,
                "adversarial_max_step": 0.25,
                "adversarial_max_turn_deg": 60.0,
            },
            "adv_source": "plan",
        },
        {
            "key": "unconditional_diffusion",
            "opponent_mode": "diffusion",
            "robot2_global": "diffusion",
            "robot2_local": "pid",
            "goal_offset": 0.0,
            "nomad_overrides": {
                "weights_path": UNCOND_WEIGHTS,
                "config_path": UNCOND_CONFIG,
                "classical_guidance_weight": 0.25,
                "opponent_guidance_enabled": True,
                "adversarial_max_step": 0.25,
                "adversarial_max_turn_deg": 60.0,
            },
            "adv_source": "plan",
        },
        {
            "key": "rule_based",
            "opponent_mode": "rule_based",
            "robot2_global": args.robot2_classical_global_planner,
            "robot2_local": "pid",
            "goal_offset": 0.0,
            "nomad_overrides": None,
            "adv_source": "odom",
        },
    ]

    combined_rows = []
    all_paths = {}
    original_user_cfg = USER_CONFIG.read_text(encoding="utf-8")

    try:
        for method in methods:
            key = method["key"]
            print(f"\n[run] {key}")
            cfg = json.loads(json.dumps(base_cfg))
            cfg["env"]["robot1_mode"] = "movebase"
            cfg["env"]["debug"] = False
            cfg["env"]["opponent"]["enabled"] = True
            cfg["env"]["opponent"]["mode"] = method["opponent_mode"]
            cfg["env"]["opponent"]["goal_offset"] = float(method["goal_offset"])
            cfg_path = out_dir / "configs" / f"{key}.yaml"
            write_yaml(cfg_path, cfg)

            set_robot_planners(
                robot1_global="theta_star",
                robot1_local="pid",
                robot2_global=method["robot2_global"],
                robot2_local=method["robot2_local"],
            )

            stack_log = out_dir / "logs" / f"{key}_stack.log"
            eval_log = out_dir / "logs" / f"{key}_eval.log"
            nomad_log = out_dir / "logs" / f"{key}_nomad.log"
            ego_plan_yaml = out_dir / "captures" / f"{key}_ego_path.yaml"
            adv_plan_yaml = out_dir / "captures" / f"{key}_adv_path.yaml"
            csv_path = out_dir / "csv" / f"{key}.csv"
            traj_dir = out_dir / "trajectories" / key

            proc = None
            stack_log_f = None
            nomad_proc = None
            nomad_log_f = None
            ego_cap = None
            adv_cap = None
            temp_nomad_params = None
            try:
                proc, stack_log_f = start_stack(stack_log)
                if not wait_for_stack_ready(180):
                    raise RuntimeError(f"stack not ready for {key}")

                if method["opponent_mode"] == "diffusion":
                    fd, tmp_name = tempfile.mkstemp(prefix=f"{key}_nomad_", suffix=".yaml")
                    os.close(fd)
                    temp_nomad_params = Path(tmp_name)
                    make_nomad_params(Path(NOMAD_SERVICE_PARAMS), method["nomad_overrides"], temp_nomad_params)
                    nomad_proc, nomad_log_f = start_nomad_service(nomad_log, params_path=temp_nomad_params)
                    if not wait_for_ros_service("/nomad/make_plan", 120):
                        raise RuntimeError(f"/nomad/make_plan not ready for {key}")

                ego_cap = start_capture("/robot1/move_base/PathPlanner/plan", ego_plan_yaml, timeout_sec=90)
                if method["adv_source"] == "plan":
                    adv_cap = start_capture("/robot2/move_base/PathPlanner/plan", adv_plan_yaml, timeout_sec=90)

                eval_cmd = (
                    "source /opt/ros/noetic/setup.bash && "
                    "source /data/lzq/miniconda3/etc/profile.d/conda.sh && "
                    "conda activate rl && "
                    f"source {PROJECT_ROOT / 'devel/setup.bash'} && "
                    f"python {EVAL_SCRIPT} "
                    f"--config {cfg_path} "
                    f"--robot1_mode movebase "
                    f"--opponent_mode {method['opponent_mode']} "
                    f"--episodes 1 "
                    f"--max_steps {args.max_steps} "
                    f"--goal_x {args.goal_x} --goal_y {args.goal_y} "
                    f"--opponent_goal_offset {method['goal_offset']} "
                    f"--episode_setup_json {setup_json} "
                    f"--random_seed {args.random_seed} "
                    f"--csv_path {csv_path} "
                    f"--record_trajectory_output_dir {traj_dir} --record_all_trajectory_episodes"
                )
                rc = run_bash_live(eval_cmd, eval_log)
                if rc != 0:
                    raise RuntimeError(f"eval failed for {key}, rc={rc}")

                time.sleep(2)
                stop_proc(ego_cap)
                stop_proc(adv_cap)
                time.sleep(1)

                ego_pts = parse_path_yaml(ego_plan_yaml)
                if not ego_pts:
                    raise RuntimeError(f"failed to capture ego global path for {key}")

                if "ego_movebase" not in all_paths:
                    all_paths["ego_movebase"] = ego_pts
                    save_points_csv(out_dir / "paths" / "ego_movebase_global_plan.csv", ego_pts, "ego_movebase", "robot1", "plan")
                    append_combined(combined_rows, ego_pts, "ego_movebase", "robot1", "plan")

                if method["adv_source"] == "plan":
                    adv_pts = parse_path_yaml(adv_plan_yaml)
                    if not adv_pts:
                        raise RuntimeError(f"failed to capture opponent plan for {key}")
                    source_type = "plan"
                else:
                    traj_csv = traj_dir / "episode_001" / "trajectory.csv"
                    adv_pts = read_eval_traj_adv(traj_csv)
                    if not adv_pts:
                        raise RuntimeError(f"failed to read rule-based odom path for {key}")
                    source_type = "odom"

                all_paths[key] = adv_pts
                save_points_csv(out_dir / "paths" / f"{key}_opponent_{source_type}.csv", adv_pts, key, "robot2", source_type)
                append_combined(combined_rows, adv_pts, key, "robot2", source_type)

            finally:
                stop_proc(ego_cap)
                stop_proc(adv_cap)
                stop_nomad_service(nomad_proc)
                if nomad_log_f is not None:
                    nomad_log_f.close()
                stop_stack(proc)
                if stack_log_f is not None:
                    stack_log_f.close()
                if temp_nomad_params is not None and temp_nomad_params.exists():
                    temp_nomad_params.unlink()

        combined_csv = out_dir / "paths" / "combined_overlay_paths.csv"
        with open(combined_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "agent", "source_type", "point_idx", "x", "y"])
            writer.writerows(combined_rows)

        plot_overlay(all_paths, out_dir / "paths" / "fixed_goal_global_plan_overlay.png", (args.goal_x, args.goal_y))
        summary = {
            "goal": [args.goal_x, args.goal_y],
            "fixed_setup_json": str(setup_json),
            "output_dir": str(out_dir),
            "path_keys": list(all_paths.keys()),
            "note": "rule_based uses executed odom path because this baseline has no global planner output topic.",
        }
        (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] results saved to {out_dir}")
    finally:
        USER_CONFIG.write_text(original_user_cfg, encoding="utf-8")


if __name__ == "__main__":
    main()
