#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path("/data/lzq/ros_motion_planning")
DEFAULT_RECOMMENDED = PROJECT_ROOT / "src/rl_training/model_screening/recommended_models.json"
BASE_CONFIG = PROJECT_ROOT / "src/rl_training/config/forklift_movebase.yaml"
EVAL_SCRIPT = PROJECT_ROOT / "src/rl_training/eval_velodyne_td3_with_goal.py"
MAIN_SCRIPT = PROJECT_ROOT / "scripts/main.sh"
KILL_SCRIPT = PROJECT_ROOT / "scripts/killpro.sh"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_yaml(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def build_eval_config(base_cfg, enabled: bool, mode: str):
    cfg = json.loads(json.dumps(base_cfg))
    env = cfg["env"]
    opponent = dict(env.get("opponent", {}))
    opponent["enabled"] = bool(enabled)
    opponent["mode"] = mode
    env["opponent"] = opponent
    return cfg


def parse_summary(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("row_type") == "summary":
                return row
    return None


def dump_summaries(path: Path, summaries):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


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
    ]:
        _run_bash(f"pkill -f '{pattern}' >/dev/null 2>&1 || true")
    time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description="Run independent validation for selected TD3 models.")
    parser.add_argument("--recommended-json", type=str, default=str(DEFAULT_RECOMMENDED))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "src/rl_training/model_screening/independent_eval"),
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument(
        "--categories",
        type=str,
        default="no_adversarial,rule_adversarial,planner_adversarial,ours_diffusion",
        help="Comma separated categories to evaluate.",
    )
    parser.add_argument("--top-per-category", type=int, default=1)
    parser.add_argument(
        "--scenarios",
        type=str,
        default="adv,noadv",
        help="Comma separated scenarios: adv,noadv",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run the evaluation commands.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stack_logs_dir = out_dir / "stack_logs"
    stack_logs_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(open(BASE_CONFIG, "r", encoding="utf-8"))
    recommended = load_json(Path(args.recommended_json))
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]

    adv_cfg_path = cfg_dir / "forklift_eval_adv.yaml"
    noadv_cfg_path = cfg_dir / "forklift_eval_noadv.yaml"
    write_yaml(adv_cfg_path, build_eval_config(base_cfg, True, "movebase"))
    write_yaml(noadv_cfg_path, build_eval_config(base_cfg, False, "movebase"))

    jobs = []
    for category in categories:
        items = recommended.get("recommended", {}).get(category, [])[: args.top_per_category]
        for item in items:
            for scenario in scenarios:
                cfg_path = adv_cfg_path if scenario == "adv" else noadv_cfg_path
                csv_path = csv_dir / f"{category}__{item['group'].replace('/', '__')}__{item['run_name']}__{scenario}.csv"
                jobs.append(
                    {
                        "category": category,
                        "scenario": scenario,
                        "config_path": str(cfg_path),
                        "group": item["group"],
                        "run_name": item["run_name"],
                        "model_path": str(PROJECT_ROOT / "src/rl_training" / item["model_path"]),
                        "csv_path": str(csv_path),
                    }
                )

    commands = []
    for job in jobs:
        cmd = (
            "source /opt/ros/noetic/setup.bash && "
            "source /data/lzq/miniconda3/etc/profile.d/conda.sh && "
            "conda activate rl && "
            f"source {PROJECT_ROOT / 'devel/setup.bash'} && "
            f"python {EVAL_SCRIPT} "
            f"--config {job['config_path']} "
            f"--model_path {job['model_path']} "
            f"--episodes {args.episodes} "
            f"--csv_path {job['csv_path']}"
        )
        commands.append({**job, "command": cmd})

    commands_path = out_dir / "validation_commands.json"
    with open(commands_path, "w", encoding="utf-8") as f:
        json.dump(commands, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {commands_path}")

    if not args.execute:
        print("[info] --execute not set; commands prepared only.")
        return

    summaries = []
    summary_json = out_dir / "independent_eval_summary.json"
    for job in commands:
        csv_path = Path(job["csv_path"])
        if csv_path.exists():
            csv_path.unlink()
        stack_log = stack_logs_dir / f"{job['category']}__{job['group'].replace('/', '__')}__{job['run_name']}__{job['scenario']}.log"
        proc = None
        log_f = None
        print(f"[run] {job['category']} {job['group']}/{job['run_name']} scenario={job['scenario']} (restart main.sh)")
        try:
            proc, log_f = start_stack(stack_log)
            if not wait_for_stack_ready():
                raise RuntimeError(f"stack not ready in time, see {stack_log}")
            subprocess.run(
                ["bash", "-lc", job["command"]],
                cwd=str(PROJECT_ROOT),
                check=True,
            )
            summary = parse_summary(csv_path)
            summaries.append({**job, "summary": summary, "stack_log": str(stack_log)})
            dump_summaries(summary_json, summaries)
        finally:
            if log_f is not None:
                log_f.flush()
            stop_stack(proc)
            if log_f is not None:
                log_f.close()
            time.sleep(2)

    dump_summaries(summary_json, summaries)
    print(f"[done] wrote {summary_json}")


if __name__ == "__main__":
    main()
