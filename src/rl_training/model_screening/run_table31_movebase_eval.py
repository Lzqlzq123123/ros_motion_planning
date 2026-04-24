#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import time
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path("/data/lzq/ros_motion_planning")
BASE_CONFIG = PROJECT_ROOT / "src/rl_training/config/forklift_movebase.yaml"
EVAL_SCRIPT = PROJECT_ROOT / "src/rl_training/eval_velodyne_td3_with_goal.py"
MAIN_SCRIPT = PROJECT_ROOT / "scripts/main.sh"
KILL_SCRIPT = PROJECT_ROOT / "scripts/killpro.sh"
USER_CONFIG = PROJECT_ROOT / "src/user_config/user_config.yaml"
VISUALNAV_ROOT = Path("/data/lzq/visualnav-transformer")
NOMAD_SERVICE_SCRIPT = VISUALNAV_ROOT / "deployment/src/nomad_plan_service.py"
NOMAD_SERVICE_PARAMS = VISUALNAV_ROOT / "deployment/config/nomad_service_params.yaml"


def write_yaml(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_summary(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_type") == "summary":
                return row
    return None


def dump_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_eval_config(base_cfg, opponent_mode: str):
    cfg = json.loads(json.dumps(base_cfg))
    env = cfg["env"]
    env["robot1_mode"] = "movebase"
    env["debug"] = False
    opponent = dict(env.get("opponent", {}))
    opponent["enabled"] = True
    opponent["mode"] = opponent_mode
    env["opponent"] = opponent
    return cfg


def deep_set(mapping, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = mapping
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_user_config():
    with open(USER_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_user_config(cfg):
    with open(USER_CONFIG, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def set_robot_planners(*, robot1_global="theta_star", robot1_local="pid", robot2_global="theta_star", robot2_local="pid"):
    cfg = load_user_config()
    robots = cfg.get("robots_config", [])
    for item in robots:
        if "robot1_global_planner" in item:
            item["robot1_global_planner"] = robot1_global
        if "robot1_local_planner" in item:
            item["robot1_local_planner"] = robot1_local
        if "robot2_global_planner" in item:
            item["robot2_global_planner"] = robot2_global
        if "robot2_local_planner" in item:
            item["robot2_local_planner"] = robot2_local
    save_user_config(cfg)


def _run_bash(command: str, check=False):
    return subprocess.run(
        ["bash", "-lc", command],
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
        "/gazebo/set_model_state",
        "/gazebo/pause_physics",
        "/gazebo/unpause_physics",
        "/robot1/odom",
        "/robot2/odom",
        "/robot1/move_base_simple/goal",
        "/robot2/move_base_simple/goal",
    ]
    return all(x in text for x in required)


def wait_for_stack_ready(timeout_sec=180):
    start = time.time()
    while time.time() - start < timeout_sec:
        if stack_ready():
            return True
        time.sleep(2)
    return False


def ros_service_ready(service_name: str):
    probe = _run_bash(
        "source /opt/ros/noetic/setup.bash && "
        "rosservice list 2>/dev/null"
    )
    if probe.returncode != 0:
        return False
    return service_name in probe.stdout.splitlines()


def wait_for_ros_service(service_name: str, timeout_sec=120):
    start = time.time()
    while time.time() - start < timeout_sec:
        if ros_service_ready(service_name):
            return True
        time.sleep(2)
    return False


def start_stack(log_path: Path):
    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", "-lc", f"cd {PROJECT_ROOT / 'scripts'} && bash {MAIN_SCRIPT}"],
        cwd=str(PROJECT_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    return proc, log_f


def stop_stack(proc):
    try:
        _run_bash(f"cd {PROJECT_ROOT / 'scripts'} && bash {KILL_SCRIPT} >/dev/null 2>&1 || true")
    except Exception:
        pass
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pattern in [
        "roslaunch sim_env main.launch",
        "gzserver",
        "gzclient",
        "roscore",
        "rosmaster",
        "move_base",
    ]:
        _run_bash(f"pkill -f '{pattern}' >/dev/null 2>&1 || true")
    time.sleep(3)


def start_nomad_service(log_path: Path, params_path: Path = None):
    log_f = open(log_path, "w", encoding="utf-8")
    ros_py = "/opt/ros/noetic/lib/python3/dist-packages"
    ws_py = str(PROJECT_ROOT / "devel/lib/python3/dist-packages")
    effective_params = params_path if params_path is not None else NOMAD_SERVICE_PARAMS
    cmd = (
        "source /opt/ros/noetic/setup.bash && "
        f"source {PROJECT_ROOT / 'devel/setup.bash'} && "
        "source /data/lzq/miniconda3/etc/profile.d/conda.sh && "
        "conda activate nomad_train && "
        "export PYTHONNOUSERSITE=1 && "
        f"export PYTHONPATH={ros_py}:{ws_py}:$PYTHONPATH && "
        f"export VISUALNAV_ROOT={VISUALNAV_ROOT} && "
        "python -c \"import sys; mods=['PIL','torch','torchvision','cv2','diffusers','rospy','cv_bridge']; "
        "print('[nomad-env] python=', sys.executable, flush=True); "
        "[( __import__(m), print(f'[nomad-env] import {m}: OK', flush=True)) for m in mods]\" && "
        f"exec python {NOMAD_SERVICE_SCRIPT} _params_file:={effective_params}"
    )
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=str(VISUALNAV_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    return proc, log_f


def stop_nomad_service(proc):
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    _run_bash("pkill -f 'nomad_plan_service.py' >/dev/null 2>&1 || true")
    time.sleep(2)


def float_or_none(v):
    if v is None or v == "":
        return None
    return float(v)


def main():
    parser = argparse.ArgumentParser(description="Unified table 3-1 evaluation with robot1=movebase.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "src/rl_training/model_screening/table3_movebase_eval_20260320"),
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--methods",
        type=str,
        default="rule_based,planner_based,ours_diffusion",
        help="Comma separated subset: rule_based,planner_based,ours_diffusion",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--goal-offset", type=float, default=None, help="Override env opponent.goal_offset for the selected method(s)")
    parser.add_argument("--robot1-global-planner", type=str, default=None, help="Override robot1 global planner for selected method(s)")
    parser.add_argument("--robot1-local-planner", type=str, default=None, help="Override robot1 local planner for selected method(s)")
    parser.add_argument("--robot2-global-planner", type=str, default=None, help="Override robot2 global planner for selected method(s)")
    parser.add_argument("--robot2-local-planner", type=str, default=None, help="Override robot2 local planner for selected method(s)")
    parser.add_argument("--nomad-weights-path", type=str, default=None, help="Override NoMaD weights_path for diffusion runs")
    parser.add_argument("--nomad-config-path", type=str, default=None, help="Override NoMaD/diffusion config_path for diffusion runs")
    parser.add_argument("--nomad-model-name", type=str, default=None, help="Override NoMaD model_name for diffusion runs")
    parser.add_argument("--nomad-classical-guidance-weight", type=float, default=None)
    parser.add_argument("--nomad-metric-waypoint-spacing", type=float, default=None)
    parser.add_argument("--nomad-classical-guidance-enabled", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--nomad-opponent-guidance-enabled", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--nomad-num-adversarial-samples", type=int, default=None)
    parser.add_argument("--nomad-adversarial-max-step", type=float, default=None)
    parser.add_argument("--nomad-adversarial-max-turn-deg", type=float, default=None)
    parser.add_argument("--nomad-adversarial-guidance-max-deviation", type=float, default=None)
    parser.add_argument("--nomad-adversarial-guidance-mean-deviation", type=float, default=None)
    parser.add_argument("--nomad-debug-dump-dir", type=str, default=None)
    parser.add_argument("--nomad-debug-dump-once", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--record-image-topic", type=str, default=None, help="Optional image topic to record during eval, e.g. /robot2/camera/rgb/image_raw")
    parser.add_argument("--record-image-save-rate", type=float, default=None, help="Save rate for recorded images; 0 saves every frame")
    parser.add_argument("--record-all-image-episodes", action="store_true", help="Record images for all episodes")
    parser.add_argument("--record-episode-indices", type=str, default=None, help="1-based episode list/range, e.g. 2,5-7")
    parser.add_argument("--eval-extra-args", type=str, default="", help="Extra raw CLI args appended to eval_velodyne_td3_with_goal.py")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stack_logs_dir = out_dir / "stack_logs"
    stack_logs_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(BASE_CONFIG)
    modes = [
        ("rule_based", "rule_based", "theta_star"),
        ("planner_based", "movebase", "theta_star"),
        ("ours_diffusion", "diffusion", "diffusion"),
        ("unconditional_diffusion", "diffusion", "diffusion"),
    ]
    selected = {x.strip() for x in args.methods.split(",") if x.strip()}

    jobs = []
    nomad_overrides = {}
    if args.nomad_weights_path is not None:
        nomad_overrides["weights_path"] = args.nomad_weights_path
    if args.nomad_config_path is not None:
        nomad_overrides["config_path"] = args.nomad_config_path
    if args.nomad_model_name is not None:
        nomad_overrides["model_name"] = args.nomad_model_name
    if args.nomad_classical_guidance_weight is not None:
        nomad_overrides["classical_guidance_weight"] = args.nomad_classical_guidance_weight
    if args.nomad_metric_waypoint_spacing is not None:
        nomad_overrides["metric_waypoint_spacing"] = args.nomad_metric_waypoint_spacing
    if args.nomad_classical_guidance_enabled is not None:
        nomad_overrides["classical_guidance_enabled"] = (args.nomad_classical_guidance_enabled == "true")
    if args.nomad_opponent_guidance_enabled is not None:
        nomad_overrides["opponent_guidance_enabled"] = (args.nomad_opponent_guidance_enabled == "true")
    if args.nomad_num_adversarial_samples is not None:
        nomad_overrides["num_adversarial_samples"] = args.nomad_num_adversarial_samples
    if args.nomad_adversarial_max_step is not None:
        nomad_overrides["adversarial_max_step"] = args.nomad_adversarial_max_step
    if args.nomad_adversarial_max_turn_deg is not None:
        nomad_overrides["adversarial_max_turn_deg"] = args.nomad_adversarial_max_turn_deg
    if args.nomad_adversarial_guidance_max_deviation is not None:
        nomad_overrides["adversarial_guidance_max_deviation"] = args.nomad_adversarial_guidance_max_deviation
    if args.nomad_adversarial_guidance_mean_deviation is not None:
        nomad_overrides["adversarial_guidance_mean_deviation"] = args.nomad_adversarial_guidance_mean_deviation
    if args.nomad_debug_dump_dir is not None:
        nomad_overrides["debug_dump_dir"] = args.nomad_debug_dump_dir
    if args.nomad_debug_dump_once is not None:
        nomad_overrides["debug_dump_once"] = (args.nomad_debug_dump_once == "true")

    for key, opponent_mode, robot2_global_planner in modes:
        if key not in selected:
            continue
        cfg_path = cfg_dir / f"{key}.yaml"
        cfg = build_eval_config(base_cfg, opponent_mode)
        if args.goal_offset is not None:
            cfg["env"]["goal_offset"] = float(args.goal_offset)
            if "opponent" in cfg["env"]:
                cfg["env"]["opponent"]["goal_offset"] = float(args.goal_offset)
        write_yaml(cfg_path, cfg)
        csv_path = csv_dir / f"{key}.csv"
        planner_override = args.robot2_global_planner if args.robot2_global_planner else robot2_global_planner
        local_planner_override = args.robot2_local_planner if args.robot2_local_planner else "pid"
        cmd = (
            "source /opt/ros/noetic/setup.bash && "
            "source /data/lzq/miniconda3/etc/profile.d/conda.sh && "
            "conda activate rl && "
            f"source {PROJECT_ROOT / 'devel/setup.bash'} && "
            f"python {EVAL_SCRIPT} "
            f"--config {cfg_path} "
            f"--robot1_mode movebase "
            f"--opponent_mode {opponent_mode} "
            f"--episodes {args.episodes} "
            f"--csv_path {csv_path}"
        )
        image_output_dir = out_dir / "episode_images" / key
        if args.record_image_topic:
            cmd += f" --record_image_topic {args.record_image_topic}"
            cmd += f" --record_image_output_dir {image_output_dir}"
            if args.record_image_save_rate is not None:
                cmd += f" --record_image_save_rate {args.record_image_save_rate}"
            if args.record_all_image_episodes:
                cmd += " --record_all_image_episodes"
            elif args.record_episode_indices:
                cmd += f" --record_episode_indices {args.record_episode_indices}"
        if args.eval_extra_args:
            cmd += f" {args.eval_extra_args}"
        jobs.append(
            {
                "key": key,
                "label": key,
                "opponent_mode": opponent_mode,
                "robot2_global_planner": planner_override,
                "robot2_local_planner": local_planner_override,
                "config_path": str(cfg_path),
                "csv_path": str(csv_path),
                "command": cmd,
                "goal_offset": args.goal_offset,
                "nomad_overrides": nomad_overrides if opponent_mode == "diffusion" else {},
            }
        )

    dump_json(out_dir / "validation_commands.json", jobs)
    print(f"[done] wrote {out_dir / 'validation_commands.json'}")

    if not args.execute:
        print("[info] --execute not set; commands prepared only.")
        return

    summaries = []
    original_user_cfg = USER_CONFIG.read_text(encoding="utf-8")
    try:
        for job in jobs:
            csv_path = Path(job["csv_path"])
            if csv_path.exists():
                csv_path.unlink()
            stack_log = stack_logs_dir / f"{job['key']}.log"
            nomad_log = stack_logs_dir / f"{job['key']}__nomad_service.log"
            proc = None
            log_f = None
            nomad_proc = None
            nomad_log_f = None
            temp_nomad_params = None
            print(f"[run] {job['key']} opponent_mode={job['opponent_mode']} robot1=movebase (restart main.sh)")
            try:
                set_robot_planners(
                    robot1_global=args.robot1_global_planner if args.robot1_global_planner else "theta_star",
                    robot1_local=args.robot1_local_planner if args.robot1_local_planner else "pid",
                    robot2_global=job["robot2_global_planner"],
                    robot2_local=job["robot2_local_planner"],
                )
                proc, log_f = start_stack(stack_log)
                if not wait_for_stack_ready():
                    raise RuntimeError(f"stack not ready in time, see {stack_log}")
                if job["opponent_mode"] == "diffusion":
                    params_path = None
                    if job.get("nomad_overrides"):
                        params_cfg = load_yaml(NOMAD_SERVICE_PARAMS)
                        for k, v in job["nomad_overrides"].items():
                            deep_set(params_cfg, k, v)
                        fd, tmp_name = tempfile.mkstemp(prefix="nomad_params_", suffix=".yaml")
                        os.close(fd)
                        temp_nomad_params = Path(tmp_name)
                        write_yaml(temp_nomad_params, params_cfg)
                        params_path = temp_nomad_params
                    nomad_proc, nomad_log_f = start_nomad_service(nomad_log, params_path=params_path)
                    if not wait_for_ros_service("/nomad/make_plan", timeout_sec=120):
                        raise RuntimeError(f"NoMaD service not ready in time, see {nomad_log}")
                subprocess.run(
                    ["bash", "-lc", job["command"]],
                    cwd=str(PROJECT_ROOT),
                    check=True,
                )
                summary = parse_summary(csv_path)
                summaries.append({**job, "summary": summary, "stack_log": str(stack_log)})
                dump_json(out_dir / "table3_eval_summary.json", summaries)
            finally:
                stop_nomad_service(nomad_proc)
                if nomad_log_f is not None:
                    nomad_log_f.flush()
                    nomad_log_f.close()
                if temp_nomad_params is not None and temp_nomad_params.exists():
                    temp_nomad_params.unlink()
                if log_f is not None:
                    log_f.flush()
                stop_stack(proc)
                if log_f is not None:
                    log_f.close()
                time.sleep(2)
    finally:
        USER_CONFIG.write_text(original_user_cfg, encoding="utf-8")

    table_rows = [
        [
            "method_key",
            "opponent_mode",
            "robot2_global_planner",
            "robot2_local_planner",
            "robot1_mode",
            "valid_success_rate",
            "collision_rate",
            "avg_min_ttc_s",
            "avg_frechet_distance",
            "avg_smoothness",
            "source_csv",
        ]
    ]
    for item in summaries:
        s = item.get("summary") or {}
        table_rows.append(
            [
                item["key"],
                item["opponent_mode"],
                item["robot2_global_planner"],
                item["robot2_local_planner"],
                s.get("robot1_mode"),
                s.get("success_rate"),
                s.get("collision_rate"),
                s.get("min_ttc"),
                s.get("frechet_distance"),
                s.get("smoothness"),
                item["csv_path"],
            ]
        )

    with (out_dir / "table3_eval_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(table_rows)

    pretty = {}
    for item in summaries:
        s = item.get("summary") or {}
        pretty[item["key"]] = {
            "robot1_mode": s.get("robot1_mode"),
            "opponent_mode": s.get("opponent_mode"),
            "robot2_local_planner": item["robot2_local_planner"],
            "success_percent": round(float_or_none(s.get("success_rate")) * 100, 1) if float_or_none(s.get("success_rate")) is not None else None,
            "collision_percent": round(float_or_none(s.get("collision_rate")) * 100, 1) if float_or_none(s.get("collision_rate")) is not None else None,
            "avg_min_ttc_s": round(float_or_none(s.get("min_ttc")), 3) if float_or_none(s.get("min_ttc")) is not None else None,
            "avg_frechet_distance": round(float_or_none(s.get("frechet_distance")), 3) if float_or_none(s.get("frechet_distance")) is not None else None,
            "avg_smoothness": float_or_none(s.get("smoothness")),
            "source_csv": item["csv_path"],
        }
    dump_json(out_dir / "table3_eval_pretty.json", pretty)
    print(f"[done] wrote {out_dir / 'table3_eval_summary.csv'}")
    print(f"[done] wrote {out_dir / 'table3_eval_pretty.json'}")


if __name__ == "__main__":
    main()
