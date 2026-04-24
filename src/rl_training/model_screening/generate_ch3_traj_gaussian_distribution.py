#!/usr/bin/env python3
import csv
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse
import numpy as np


DATASET_DIR = Path("/data/lzq/datasets/gazebo")
METHOD_BASE = Path(
    "/data/lzq/ros_motion_planning/src/rl_training/model_screening/fixed_goal_m3_m4_random_20260323"
)
SCRIPT_PATH = Path(__file__).resolve()

OUTPUT_DIR = Path(
    "/data/lzq/tjuthesis/figures/paper_data_audit_20260324/"
    "fig_3_gaussian_trajectory_distribution"
)
FINAL_FIGURE_DIR = Path("/data/lzq/tjuthesis/figures/diffusion")

METHODS = [
    ("rule", "规则方法"),
    ("planner", "规划器驱动方法"),
    ("unconditional", "无视觉条件扩散方法"),
    ("ours", "本文方法"),
]

COLORS = {
    "dataset": "#4d4d4d",
    "rule": "#f28e2b",
    "planner": "#4e79a7",
    "unconditional": "#b07aa1",
    "ours": "#e15759",
}

RESAMPLE_POINTS = 16
ELLIPSE_POINTS = [0, 3, 6, 9, 12, 15]
ELLIPSE_STD = 1.5


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_style():
    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/data/lzq/.local/share/fonts/windows/simsun.ttc",
    ]
    chosen_name = None
    for font_path in font_candidates:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            chosen_name = font_manager.FontProperties(fname=font_path).get_name()
            break
    plt.rcParams["font.family"] = chosen_name or "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = [chosen_name or "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10


def resample_polyline(points, n_points=RESAMPLE_POINTS):
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros((n_points, 2), dtype=float)
    if len(pts) == 1:
        return np.repeat(pts[:1], n_points, axis=0)

    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = float(cum_len[-1])
    if total_len < 1e-8:
        return np.repeat(pts[:1], n_points, axis=0)

    targets = np.linspace(0.0, total_len, n_points)
    out = []
    idx = 0
    for t in targets:
        while idx < len(seg_len) - 1 and cum_len[idx + 1] < t:
            idx += 1
        denom = max(cum_len[idx + 1] - cum_len[idx], 1e-8)
        alpha = (t - cum_len[idx]) / denom
        out.append((1.0 - alpha) * pts[idx] + alpha * pts[idx + 1])
    return np.asarray(out, dtype=float)


def align_and_normalize_shape(points):
    pts = np.asarray(points, dtype=float)
    pts = pts - pts[0]

    ref_idx = min(len(pts) - 1, 3)
    ref_vec = pts[ref_idx]
    angle = math.atan2(ref_vec[1], ref_vec[0]) if np.linalg.norm(ref_vec) > 1e-8 else 0.0
    c, s = math.cos(-angle), math.sin(-angle)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    pts = pts @ rot.T

    path_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    if path_len > 1e-8:
        pts = pts / path_len
    return pts


def load_dataset_trajectories():
    trajs = []
    for traj_file in sorted(DATASET_DIR.glob("*/traj_data.pkl")):
        with open(traj_file, "rb") as f:
            obj = pickle.load(f)
        raw = np.asarray(obj["position"], dtype=float)
        trajs.append(align_and_normalize_shape(resample_polyline(raw)))
    return np.asarray(trajs, dtype=float)


def load_method_trajectories(method_key):
    trajs = []
    traj_root = METHOD_BASE / method_key / "trajectories"
    for traj_file in sorted(traj_root.glob("episode_*/trajectory.csv")):
        raw = []
        with open(traj_file, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                raw.append([float(row["adv_x"]), float(row["adv_y"])])
        trajs.append(align_and_normalize_shape(resample_polyline(np.asarray(raw, dtype=float))))
    return np.asarray(trajs, dtype=float)


def covariance_ellipse(points, n_std=ELLIPSE_STD):
    if len(points) < 2:
        return None
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 1e-12))
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    center = points.mean(axis=0)
    return center, width, height, angle


def draw_gaussian_path(ax, trajs, color, label, edge_alpha=1.0, fill_alpha=0.08, zorder=2):
    mean_traj = trajs.mean(axis=0)
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], color=color, lw=2.2, label=label, zorder=zorder + 1)
    ax.scatter(mean_traj[0, 0], mean_traj[0, 1], color=color, s=20, zorder=zorder + 2)
    for idx in ELLIPSE_POINTS:
        desc = covariance_ellipse(trajs[:, idx, :])
        if desc is None:
            continue
        center, width, height, angle = desc
        e = Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            facecolor=matplotlib.colors.to_rgba(color, fill_alpha),
            edgecolor=matplotlib.colors.to_rgba(color, edge_alpha),
            lw=1.2,
            zorder=zorder,
        )
        ax.add_patch(e)
    return mean_traj


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ensure_dirs()
    set_plot_style()

    expert = load_dataset_trajectories()
    method_data = {k: load_method_trajectories(k) for k, _ in METHODS}

    gaussian_rows = []
    summary_rows = []
    expert_mean = expert.mean(axis=0)
    expert_cov_trace = []
    for idx in range(RESAMPLE_POINTS):
        cov = np.cov(expert[:, idx, :].T)
        expert_cov_trace.append(float(np.trace(cov)))

    for method_key, method_name in METHODS:
        trajs = method_data[method_key]
        method_mean = trajs.mean(axis=0)
        mean_gap = float(np.mean(np.linalg.norm(method_mean - expert_mean, axis=1)))
        cov_gap = 0.0
        for idx in range(RESAMPLE_POINTS):
            cov = np.cov(trajs[:, idx, :].T)
            cov_gap += abs(float(np.trace(cov)) - expert_cov_trace[idx])
            gaussian_rows.append(
                {
                    "group_key": method_key,
                    "group_name": method_name,
                    "progress_index": idx,
                    "progress_ratio": round(idx / (RESAMPLE_POINTS - 1), 6),
                    "mean_x": round(float(method_mean[idx, 0]), 6),
                    "mean_y": round(float(method_mean[idx, 1]), 6),
                    "cov_xx": round(float(cov[0, 0]), 6),
                    "cov_xy": round(float(cov[0, 1]), 6),
                    "cov_yy": round(float(cov[1, 1]), 6),
                }
            )
        summary_rows.append(
            {
                "group_name": method_name,
                "num_trajectories": len(trajs),
                "mean_path_gap_to_expert": round(mean_gap, 6),
                "covariance_trace_gap_to_expert": round(cov_gap / RESAMPLE_POINTS, 6),
            }
        )

    write_csv(
        OUTPUT_DIR / "trajectory_gaussian_params.csv",
        [
            "group_key",
            "group_name",
            "progress_index",
            "progress_ratio",
            "mean_x",
            "mean_y",
            "cov_xx",
            "cov_xy",
            "cov_yy",
        ],
        gaussian_rows,
    )
    write_csv(
        OUTPUT_DIR / "trajectory_gaussian_summary.csv",
        ["group_name", "num_trajectories", "mean_path_gap_to_expert", "covariance_trace_gap_to_expert"],
        summary_rows,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.2), dpi=220)
    axes = axes.ravel()
    for ax, (method_key, method_name) in zip(axes, METHODS):
        draw_gaussian_path(
            ax,
            expert,
            COLORS["dataset"],
            "专家轨迹分布",
            edge_alpha=0.75,
            fill_alpha=0.05,
            zorder=1,
        )
        draw_gaussian_path(
            ax,
            method_data[method_key],
            COLORS[method_key],
            method_name,
            edge_alpha=0.85,
            fill_alpha=0.10,
            zorder=3,
        )
        ax.set_title(method_name)
        ax.set_xlabel("归一化前向位移")
        ax.set_ylabel("归一化横向位移")
        ax.grid(alpha=0.15, linestyle="--")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    out_png = OUTPUT_DIR / "trajectory_gaussian_distribution_compare.png"
    out_pdf = OUTPUT_DIR / "trajectory_gaussian_distribution_compare.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    final_png = FINAL_FIGURE_DIR / "轨迹高斯分布对比.png"
    final_pdf = FINAL_FIGURE_DIR / "轨迹高斯分布对比.pdf"
    final_png.write_bytes(out_png.read_bytes())
    final_pdf.write_bytes(out_pdf.read_bytes())

    script_copy = OUTPUT_DIR / SCRIPT_PATH.name
    script_copy.write_text(SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with open(OUTPUT_DIR / "plot_method.txt", "w", encoding="utf-8") as f:
        f.write(
            "轨迹高斯分布可视化说明\n"
            "1. 所有轨迹统一重采样为 16 个点，并完成起点平移、朝向对齐和长度归一化。\n"
            "2. 每个子图中，灰色表示专家轨迹在各采样进度下的二维高斯分布；彩色表示对应方法在同一进度下的二维高斯分布。\n"
            "3. 实线表示均值轨迹，椭圆表示该进度下位置分布的协方差范围。\n"
            "4. 若某方法彩色椭圆与灰色椭圆更贴近、均值轨迹重合度更高，则说明其轨迹分布更接近专家数据分布。\n"
        )

    print(f"saved figure: {out_png}")
    print(f"saved summary: {OUTPUT_DIR / 'trajectory_gaussian_summary.csv'}")


if __name__ == "__main__":
    main()
