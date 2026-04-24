#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np


SRC_BASE = Path("/data/lzq/ros_motion_planning/src/rl_training/data")
OUT_BASE = Path(
    "/data/lzq/ros_motion_planning/src/rl_training/model_screening/"
    "paper_results_20260325/table_4_4_curriculum_q_metrics"
)
SOURCE_CSV_DIR = OUT_BASE / "source_csv"
OUT_BASE.mkdir(parents=True, exist_ok=True)
SOURCE_CSV_DIR.mkdir(exist_ok=True)

RUN_GROUPS = {
    "无课程学习训练": {
        "runs": ["run_noadv_nocurr_seed1", "run_noadv_nocurr_seed2"],
        "max_round_used": 1000,
        "table_metric_mode": "seedwise_average",
    },
    "课程学习训练": {
        "runs": ["run_1", "run_13"],
        "max_round_used": None,
        "table_metric_mode": "seedwise_average",
    },
}

SOURCE_METRIC = "Max._Q.csv"
SMOOTH_WEIGHT = 0.9
TAIL = 50
CONV_RATIO = 0.90
STABLE_HOLD_ROUNDS = 100


def load_csv_metric(csv_path: Path, max_round: Optional[int]):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            step = float(row["step"])
            value = float(row["value"])
            wall_time = float(row.get("wall_time", 0) or 0)
            rows.append((step, wall_time, value))

    rows.sort(key=lambda x: (x[0], x[1]))
    dedup = []
    for step, wall_time, value in rows:
        if dedup and dedup[-1][0] == step:
            dedup[-1] = (step, value)
        else:
            dedup.append((step, value))

    steps = np.array([x for x, _ in dedup], dtype=float)
    values = np.array([y for _, y in dedup], dtype=float)
    if max_round is not None:
        mask = steps <= max_round
        steps = steps[mask]
        values = values[mask]
    return steps, values


def ewm(values: np.ndarray, weight: float):
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = (1.0 - weight) * values[i] + weight * out[i - 1]
    return out


def compute_stable_reach_step(steps: np.ndarray, curve: np.ndarray, convergence_value: float):
    lower = convergence_value * (1.0 - (1.0 - CONV_RATIO))
    upper = convergence_value * (1.0 + (1.0 - CONV_RATIO))
    for i in range(len(curve) - STABLE_HOLD_ROUNDS + 1):
        seg = curve[i : i + STABLE_HOLD_ROUNDS]
        if np.all((seg >= lower) & (seg <= upper)):
            return float(steps[i]), float(lower), float(upper)
    return None, float(lower), float(upper)


def compute_curve_metrics(steps: np.ndarray, curve: np.ndarray):
    convergence_value = float(np.mean(curve[-TAIL:]))
    stable_reach_step, lower, upper = compute_stable_reach_step(steps, curve, convergence_value)
    peak_idx = int(np.argmax(curve))
    peak_value = float(curve[peak_idx])
    peak_step = float(steps[peak_idx])
    peak_gap = peak_value - convergence_value
    return {
        "convergence_value": convergence_value,
        "tail_variance": float(np.var(curve[-TAIL:])),
        "stable_band_lower": lower,
        "stable_band_upper": upper,
        "episodes_to_90_convergence_stable": stable_reach_step,
        "max_q_peak": peak_value,
        "max_q_peak_step": peak_step,
        "max_q_peak_gap_to_convergence": float(peak_gap),
    }


def save_group_curve_csv(label, steps, mean_curve, min_curve, max_curve):
    out_csv = OUT_BASE / f"{label}_group_curve.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_q", "min_q", "max_q"])
        for s, m, lo, hi in zip(steps, mean_curve, min_curve, max_curve):
            writer.writerow([float(s), float(m), float(lo), float(hi)])
    return str(out_csv)


def round_or_blank(value, digits=2):
    if value is None:
        return ""
    return round(float(value), digits)


def main():
    per_run_rows = []
    summary_rows = []
    detail = {}

    for label, cfg in RUN_GROUPS.items():
        max_round_used = cfg["max_round_used"]
        table_metric_mode = cfg["table_metric_mode"]

        series = []
        for run in cfg["runs"]:
            src = SRC_BASE / run / SOURCE_METRIC
            dst = SOURCE_CSV_DIR / f"{run}__{SOURCE_METRIC}"
            dst.write_text(src.read_text(encoding="utf-8-sig"), encoding="utf-8")

            steps, raw_values = load_csv_metric(src, max_round=max_round_used)
            smooth_values = ewm(raw_values, SMOOTH_WEIGHT)
            curve_metrics = compute_curve_metrics(steps, smooth_values)

            run_row = {
                "training_mode": label,
                "run": run,
                "num_points_used": int(len(steps)),
                "max_round_used": max_round_used if max_round_used is not None else "full",
                "raw_full_variance": float(np.var(raw_values)),
                "smoothed_full_variance": float(np.var(smooth_values)),
                "tail_variance": curve_metrics["tail_variance"],
                "max_q_convergence_value": curve_metrics["convergence_value"],
                "max_q_peak": curve_metrics["max_q_peak"],
                "max_q_peak_step": curve_metrics["max_q_peak_step"],
                "max_q_peak_gap_to_convergence": curve_metrics["max_q_peak_gap_to_convergence"],
                "episodes_to_90_convergence_stable": curve_metrics["episodes_to_90_convergence_stable"],
                "stable_band_lower": curve_metrics["stable_band_lower"],
                "stable_band_upper": curve_metrics["stable_band_upper"],
                "source_csv": str(dst),
            }
            per_run_rows.append(run_row)
            series.append((run, steps, smooth_values, run_row))

        common_steps = np.array(
            sorted(set(np.concatenate([steps for _, steps, _, _ in series]))), dtype=float
        )
        aligned = np.full((len(series), len(common_steps)), np.nan, dtype=float)
        for i, (_, steps, values, _) in enumerate(series):
            mask = (common_steps >= steps.min()) & (common_steps <= steps.max())
            aligned[i, mask] = np.interp(common_steps[mask], steps, values)
        valid = ~np.all(np.isnan(aligned), axis=0)
        common_steps = common_steps[valid]
        aligned = aligned[:, valid]
        mean_curve = np.nanmean(aligned, axis=0)
        min_curve = np.nanmin(aligned, axis=0)
        max_curve = np.nanmax(aligned, axis=0)
        group_curve_csv = save_group_curve_csv(label, common_steps, mean_curve, min_curve, max_curve)
        group_curve_metrics = compute_curve_metrics(common_steps, mean_curve)

        run_convergence_values = np.array(
            [row["max_q_convergence_value"] for _, _, _, row in series], dtype=float
        )
        run_peak_gaps = np.array(
            [row["max_q_peak_gap_to_convergence"] for _, _, _, row in series], dtype=float
        )
        run_stable_steps = np.array(
            [row["episodes_to_90_convergence_stable"] for _, _, _, row in series], dtype=float
        )
        run_raw_variances = np.array([row["raw_full_variance"] for _, _, _, row in series], dtype=float)
        run_smoothed_variances = np.array(
            [row["smoothed_full_variance"] for _, _, _, row in series], dtype=float
        )
        run_tail_variances = np.array([row["tail_variance"] for _, _, _, row in series], dtype=float)

        seedwise_average_metrics = {
            "convergence_value": float(np.mean(run_convergence_values)),
            "max_q_peak_gap_to_convergence": float(np.mean(run_peak_gaps)),
            "episodes_to_90_convergence_stable": float(np.mean(run_stable_steps)),
        }

        if table_metric_mode == "group_curve":
            table_metrics = {
                "max_q_convergence_value": group_curve_metrics["convergence_value"],
                "max_q_peak_gap_to_convergence": group_curve_metrics[
                    "max_q_peak_gap_to_convergence"
                ],
                "episodes_to_90_convergence_stable": group_curve_metrics[
                    "episodes_to_90_convergence_stable"
                ],
            }
        elif table_metric_mode == "seedwise_average":
            table_metrics = {
                "max_q_convergence_value": seedwise_average_metrics["convergence_value"],
                "max_q_peak_gap_to_convergence": seedwise_average_metrics[
                    "max_q_peak_gap_to_convergence"
                ],
                "episodes_to_90_convergence_stable": seedwise_average_metrics[
                    "episodes_to_90_convergence_stable"
                ],
            }
        else:
            raise ValueError(f"Unsupported table_metric_mode: {table_metric_mode}")

        summary_rows.append(
            {
                "training_mode": label,
                "table_metric_mode": table_metric_mode,
                "max_q_convergence_value": round_or_blank(
                    table_metrics["max_q_convergence_value"], 2
                ),
                "max_q_peak_gap_to_convergence": round_or_blank(
                    table_metrics["max_q_peak_gap_to_convergence"], 2
                ),
                "episodes_to_90_convergence_stable": round_or_blank(
                    table_metrics["episodes_to_90_convergence_stable"], 1
                ),
                "group_curve_convergence_value": round_or_blank(
                    group_curve_metrics["convergence_value"], 2
                ),
                "group_curve_peak_gap": round_or_blank(
                    group_curve_metrics["max_q_peak_gap_to_convergence"], 2
                ),
                "group_curve_stable_episode": round_or_blank(
                    group_curve_metrics["episodes_to_90_convergence_stable"], 1
                ),
                "seedwise_mean_convergence_value": round_or_blank(
                    seedwise_average_metrics["convergence_value"], 2
                ),
                "seedwise_mean_peak_gap": round_or_blank(
                    seedwise_average_metrics["max_q_peak_gap_to_convergence"], 2
                ),
                "seedwise_mean_stable_episode": round_or_blank(
                    seedwise_average_metrics["episodes_to_90_convergence_stable"], 1
                ),
                "convergence_value_across_runs_variance": round_or_blank(
                    np.var(run_convergence_values), 2
                ),
                "convergence_value_across_runs_std": round_or_blank(
                    np.std(run_convergence_values), 2
                ),
                "peak_gap_across_runs_variance": round_or_blank(np.var(run_peak_gaps), 2),
                "peak_gap_across_runs_std": round_or_blank(np.std(run_peak_gaps), 2),
                "stable_episode_across_runs_variance": round_or_blank(
                    np.var(run_stable_steps), 2
                ),
                "stable_episode_across_runs_std": round_or_blank(np.std(run_stable_steps), 2),
                "raw_full_variance_mean": round_or_blank(np.mean(run_raw_variances), 2),
                "raw_full_variance_across_runs_variance": round_or_blank(
                    np.var(run_raw_variances), 2
                ),
                "smoothed_full_variance_mean": round_or_blank(np.mean(run_smoothed_variances), 2),
                "smoothed_full_variance_across_runs_variance": round_or_blank(
                    np.var(run_smoothed_variances), 2
                ),
                "tail_variance_mean": round_or_blank(np.mean(run_tail_variances), 2),
                "tail_variance_across_runs_variance": round_or_blank(
                    np.var(run_tail_variances), 2
                ),
                "tail_window_points": TAIL,
                "smooth_weight": SMOOTH_WEIGHT,
                "max_round_used": max_round_used if max_round_used is not None else "full",
                "source_metric": SOURCE_METRIC,
                "stable_hold_rounds": STABLE_HOLD_ROUNDS,
            }
        )

        detail[label] = {
            "runs": [
                {
                    "run": row["run"],
                    "num_points_used": row["num_points_used"],
                    "max_round_used": row["max_round_used"],
                    "source_csv": row["source_csv"],
                    "raw_full_variance": row["raw_full_variance"],
                    "smoothed_full_variance": row["smoothed_full_variance"],
                    "tail_variance": row["tail_variance"],
                    "max_q_convergence_value": row["max_q_convergence_value"],
                    "max_q_peak": row["max_q_peak"],
                    "max_q_peak_step": row["max_q_peak_step"],
                    "max_q_peak_gap_to_convergence": row["max_q_peak_gap_to_convergence"],
                    "episodes_to_90_convergence_stable": row[
                        "episodes_to_90_convergence_stable"
                    ],
                    "stable_band_lower": row["stable_band_lower"],
                    "stable_band_upper": row["stable_band_upper"],
                }
                for _, _, _, row in series
            ],
            "table_metric_mode": table_metric_mode,
            "table_metrics": table_metrics,
            "group_curve_csv": group_curve_csv,
            "group_curve_metrics": group_curve_metrics,
            "seedwise_average_metrics": seedwise_average_metrics,
            "across_runs_variance": {
                "convergence_value_variance": float(np.var(run_convergence_values)),
                "peak_gap_variance": float(np.var(run_peak_gaps)),
                "stable_episode_variance": float(np.var(run_stable_steps)),
                "raw_full_variance_variance": float(np.var(run_raw_variances)),
                "smoothed_full_variance_variance": float(np.var(run_smoothed_variances)),
                "tail_variance_variance": float(np.var(run_tail_variances)),
            },
            "tail_window_points": TAIL,
            "smooth_weight": SMOOTH_WEIGHT,
            "source_metric": SOURCE_METRIC,
            "stable_hold_rounds": STABLE_HOLD_ROUNDS,
        }

    summary_csv = OUT_BASE / "table_4_4_curriculum_q_metrics_summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    per_run_csv = OUT_BASE / "table_4_4_curriculum_q_metrics_all_runs.csv"
    with open(per_run_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

    with open(
        OUT_BASE / "table_4_4_curriculum_q_metrics_detail.json", "w", encoding="utf-8"
    ) as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    with open(OUT_BASE / "metric_definition.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_metric": SOURCE_METRIC,
                "data_basis": (
                    "无课程学习组仅使用前1000回合，课程学习组使用完整训练序列；"
                    "两组均先对每条 Max.Q 曲线做指数滑动平均(weight=0.9)。"
                ),
                "max_q_convergence_value": "单次运行末尾50个点均值；表中各训练组均对组内 seeds 的单次结果做算术平均",
                "max_q_peak_gap_to_convergence": "单次运行峰值 Max.Q 与该运行最终收敛均值之差，表示最大超调波动幅度；表中各训练组均对组内 seeds 的单次结果做算术平均",
                "episodes_to_90_convergence_stable": "单次运行进入最终收敛均值±10%区间后，连续100个回合保持稳定时的起始回合数；表中各训练组均对组内 seeds 的单次结果做算术平均",
                "all_variance_saved": "summary.csv 与 all_runs.csv 额外保存了跨运行方差、全序列方差与尾部方差，便于后续审查",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(summary_csv.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
