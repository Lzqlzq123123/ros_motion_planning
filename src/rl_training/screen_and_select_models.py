#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import yaml


MANUAL_CATEGORY_MAP = {
    ("forklift_movebase", "run_9"): "planner_adversarial",
    ("forklift_movebase", "run_10"): "rule_adversarial",
    ("forklift_movebase_with_goal", "run_1"): "ours_diffusion",
    ("forklift_movebase_with_goal", "run_2"): "ours_diffusion",
}


def safe_load_yaml(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_history(path: Path):
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True).astype(float)


def infer_category(group_name: str, run_name: str, cfg):
    if (group_name, run_name) in MANUAL_CATEGORY_MAP:
        return MANUAL_CATEGORY_MAP[(group_name, run_name)]

    if cfg is not None:
        opponent = cfg.get("opponent", {})
        enabled = bool(opponent.get("enabled", False))
        mode = opponent.get("mode", "none")
        if not enabled:
            return "no_adversarial"
        if mode == "rule_based":
            return "rule_adversarial"
        return "planner_adversarial"

    if "noadv" in run_name.lower():
        return "no_adversarial"
    return "unknown"


def normalize_success(success_history, eval_episodes_hint):
    if success_history is None or len(success_history) == 0:
        return success_history, "unknown"
    arr = success_history.astype(float)
    if np.nanmax(arr) <= 1.000001:
        return arr, "ratio_0_1"
    denom = float(eval_episodes_hint or 10)
    return arr / denom, f"count_0_{int(denom)}"


def mean_last(arr, k=10):
    if arr is None or len(arr) == 0:
        return None
    return float(np.mean(arr[-min(k, len(arr)) :]))


def rel_model_path(run_dir: Path, logs_root: Path):
    try:
        rel = run_dir.relative_to(logs_root)
    except ValueError:
        rel = run_dir
    return f"logs/{rel.as_posix()}/td3_model"


def scan_eval_csv(eval_csv_dir: Path):
    summary_map = {}
    if not eval_csv_dir.exists():
        return summary_map

    for csv_file in sorted(eval_csv_dir.glob("*.csv")):
        with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("row_type") != "summary":
                    continue
                model_path = (row.get("model_path") or "").strip()
                if not model_path:
                    continue
                entry = {
                    "csv_file": csv_file.name,
                    "opponent_enabled": row.get("opponent_enabled"),
                    "opponent_mode": row.get("opponent_mode"),
                    "avg_reward": row.get("avg_reward"),
                    "success_rate": row.get("success_rate"),
                    "collision_rate": row.get("collision_rate"),
                    "timeout_rate": row.get("timeout_rate"),
                }
                summary_map.setdefault(model_path, []).append(entry)
    return summary_map


def pick_top(records, category, sort_key, top_k=2):
    subset = [r for r in records if r["category"] == category]
    subset = [r for r in subset if r.get(sort_key) is not None]
    subset.sort(key=lambda r: (r.get(sort_key, -1), r.get("final_success_rate", -1)), reverse=True)
    return subset[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Screen TD3 model runs and recommend candidates.")
    parser.add_argument(
        "--logs-root",
        type=str,
        default="/data/lzq/ros_motion_planning/src/rl_training/logs",
    )
    parser.add_argument(
        "--eval-csv-dir",
        type=str,
        default="/data/lzq/ros_motion_planning/src/rl_training/eval_csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/lzq/ros_motion_planning/src/rl_training/model_screening",
    )
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    logs_root = Path(args.logs_root)
    eval_csv_dir = Path(args.eval_csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    external_eval_map = scan_eval_csv(eval_csv_dir)
    records = []

    for actor_path in sorted(logs_root.rglob("td3_model_actor.pth")):
        run_dir = actor_path.parent
        run_name = run_dir.name
        group_rel = run_dir.parent.relative_to(logs_root).as_posix()
        group_name = run_dir.parent.name

        cfg = safe_load_yaml(run_dir / "experiment_config.yaml")
        eval_episodes_hint = None
        opponent_enabled = None
        opponent_mode = None
        seed = None
        config_present = cfg is not None
        if cfg is not None:
            eval_episodes_hint = (
                cfg.get("hyperparameters", {}).get("eval_episodes")
                or cfg.get("training", {}).get("eval_episodes")
                or 10
            )
            opponent_enabled = cfg.get("opponent", {}).get("enabled")
            opponent_mode = cfg.get("opponent", {}).get("mode")
            seed = cfg.get("runner", {}).get("seed")

        category = infer_category(group_name, run_name, cfg)

        success_hist_raw = load_history(run_dir / "eval_success_history.npy")
        success_hist, success_scale = normalize_success(success_hist_raw, eval_episodes_hint)
        collision_hist = load_history(run_dir / "eval_col_history.npy")
        reward_hist = load_history(run_dir / "eval_reward_history.npy")

        if success_hist is None:
            continue

        if collision_hist is None:
            collision_hist = np.zeros_like(success_hist)
        if reward_hist is None:
            reward_hist = np.zeros_like(success_hist)

        min_len = min(len(success_hist), len(collision_hist), len(reward_hist))
        success_hist = success_hist[:min_len]
        collision_hist = collision_hist[:min_len]
        reward_hist = reward_hist[:min_len]
        score_hist = success_hist - 0.5 * collision_hist

        model_path = rel_model_path(run_dir, logs_root)
        ext_evals = external_eval_map.get(model_path, [])

        record = {
            "group": group_rel,
            "run_name": run_name,
            "run_dir": str(run_dir),
            "model_path": model_path,
            "config_present": config_present,
            "category": category,
            "seed": seed,
            "opponent_enabled": opponent_enabled,
            "opponent_mode": opponent_mode,
            "n_eval_points": int(len(success_hist)),
            "success_scale": success_scale,
            "best_success_rate": float(np.max(success_hist)),
            "final_success_rate": float(success_hist[-1]),
            "last10_success_rate": mean_last(success_hist, 10),
            "best_collision_rate": float(np.min(collision_hist)),
            "final_collision_rate": float(collision_hist[-1]),
            "last10_collision_rate": mean_last(collision_hist, 10),
            "best_score": float(np.max(score_hist)),
            "final_score": float(score_hist[-1]),
            "last10_score": mean_last(score_hist, 10),
            "best_reward": float(np.max(reward_hist)),
            "final_reward": float(reward_hist[-1]),
            "last10_reward": mean_last(reward_hist, 10),
            "external_eval_count": len(ext_evals),
            "external_evals": ext_evals,
        }
        records.append(record)

    records.sort(key=lambda r: (r["category"], -(r["last10_score"] if r["last10_score"] is not None else -999)))

    csv_path = output_dir / "model_screening_summary.csv"
    json_path = output_dir / "model_screening_summary.json"
    rec_path = output_dir / "recommended_models.json"

    fieldnames = [
        "group",
        "run_name",
        "category",
        "seed",
        "config_present",
        "opponent_enabled",
        "opponent_mode",
        "n_eval_points",
        "success_scale",
        "best_success_rate",
        "final_success_rate",
        "last10_success_rate",
        "best_collision_rate",
        "final_collision_rate",
        "last10_collision_rate",
        "best_score",
        "final_score",
        "last10_score",
        "best_reward",
        "final_reward",
        "last10_reward",
        "external_eval_count",
        "model_path",
        "run_dir",
    ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fieldnames})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    recommendations = {
        "selection_rule": "按 final checkpoint 的 last10_score 优先筛选；若已有独立验证结果，可结合 external_evals 二次确认。",
        "top_k": args.top_k,
        "recommended": {
            "no_adversarial": pick_top(records, "no_adversarial", "last10_score", args.top_k),
            "rule_adversarial": pick_top(records, "rule_adversarial", "last10_score", args.top_k),
            "planner_adversarial": pick_top(records, "planner_adversarial", "last10_score", args.top_k),
            "ours_diffusion": pick_top(records, "ours_diffusion", "last10_score", args.top_k),
            "unknown": pick_top(records, "unknown", "last10_score", args.top_k),
        },
    }
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)

    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {rec_path}")
    print()
    for category, items in recommendations["recommended"].items():
        if not items:
            continue
        print(f"[{category}]")
        for item in items:
            print(
                f"  - {item['group']}/{item['run_name']}: "
                f"last10_success={item['last10_success_rate']:.3f}, "
                f"last10_collision={item['last10_collision_rate']:.3f}, "
                f"last10_score={item['last10_score']:.3f}, "
                f"external_eval_count={item['external_eval_count']}"
            )
        print()


if __name__ == "__main__":
    main()
