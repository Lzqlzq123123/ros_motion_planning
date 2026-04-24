#!/usr/bin/env python3
import csv
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


DATASET_DIR = Path("/data/lzq/datasets/gazebo")
METHOD_BASE = Path(
    "/data/lzq/ros_motion_planning/src/rl_training/model_screening/fixed_goal_m3_m4_random_20260323"
)
SCRIPT_PATH = Path(__file__).resolve()

OUTPUT_DIR = Path(
    "/data/lzq/tjuthesis/figures/paper_data_audit_20260324/"
    "fig_3_gaussian_1d_distribution"
)
FINAL_FIGURE_DIR = Path("/data/lzq/tjuthesis/figures/diffusion")

METHODS = [
    ("dataset", "数据集专家轨迹"),
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
    plt.rcParams["font.size"] = 11


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


def gaussian_pdf(x, mu, sigma):
    sigma = max(float(sigma), 1e-6)
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ensure_dirs()
    set_plot_style()

    expert = load_dataset_trajectories()
    expert_mean_traj = expert.mean(axis=0)

    per_group_distance = {}
    summary_rows = []
    sample_rows = []

    for method_key, method_name in METHODS:
        if method_key == "dataset":
            trajs = expert
        else:
            trajs = load_method_trajectories(method_key)

        # 每条轨迹与专家均值轨迹的平均点距
        dists = np.linalg.norm(trajs - expert_mean_traj[None, :, :], axis=2).mean(axis=1)
        per_group_distance[method_key] = dists

        mu = float(np.mean(dists))
        sigma = float(np.std(dists))
        summary_rows.append(
            {
                "group_key": method_key,
                "group_name": method_name,
                "num_trajectories": len(dists),
                "distance_mean": round(mu, 6),
                "distance_std": round(sigma, 6),
                "distance_median": round(float(np.median(dists)), 6),
                "distance_min": round(float(np.min(dists)), 6),
                "distance_max": round(float(np.max(dists)), 6),
            }
        )
        for i, d in enumerate(dists):
            sample_rows.append(
                {
                    "group_key": method_key,
                    "group_name": method_name,
                    "sample_index": i,
                    "normalized_offset_distance": round(float(d), 6),
                }
            )

    write_csv(
        OUTPUT_DIR / "gaussian_1d_summary.csv",
        [
            "group_key",
            "group_name",
            "num_trajectories",
            "distance_mean",
            "distance_std",
            "distance_median",
            "distance_min",
            "distance_max",
        ],
        summary_rows,
    )
    write_csv(
        OUTPUT_DIR / "gaussian_1d_samples.csv",
        ["group_key", "group_name", "sample_index", "normalized_offset_distance"],
        sample_rows,
    )

    x_min = min(float(np.min(v)) for v in per_group_distance.values())
    x_max = max(float(np.max(v)) for v in per_group_distance.values())
    x_pad = 0.08 * (x_max - x_min + 1e-8)
    x = np.linspace(max(0.0, x_min - x_pad), x_max + x_pad, 600)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=240)
    for row in summary_rows:
        key = row["group_key"]
        label = row["group_name"]
        mu = row["distance_mean"]
        sigma = row["distance_std"]
        color = COLORS[key]
        y = gaussian_pdf(x, mu, sigma)
        lw = 2.8 if key in ("dataset", "ours") else 2.0
        alpha = 0.95 if key in ("dataset", "ours") else 0.75
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, label=label)
        ax.axvline(mu, color=color, lw=1.0, alpha=0.35, linestyle="--")

    ax.set_xlabel("归一化偏移距离")
    ax.set_ylabel("概率密度")
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()

    out_png = OUTPUT_DIR / "gaussian_1d_distribution.png"
    out_pdf = OUTPUT_DIR / "gaussian_1d_distribution.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    final_png = FINAL_FIGURE_DIR / "轨迹高斯分布对比_一维.png"
    final_pdf = FINAL_FIGURE_DIR / "轨迹高斯分布对比_一维.pdf"
    final_png.write_bytes(out_png.read_bytes())
    final_pdf.write_bytes(out_pdf.read_bytes())

    (OUTPUT_DIR / SCRIPT_PATH.name).write_text(SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    with open(OUTPUT_DIR / "plot_method.txt", "w", encoding="utf-8") as f:
        f.write(
            "一维高斯分布图说明\n"
            "1. 先对所有轨迹进行重采样、起点平移、朝向对齐和长度归一化。\n"
            "2. 对每条轨迹计算其相对专家均值轨迹的平均点距，记作归一化偏移距离。\n"
            "3. 对各方法的偏移距离样本分别估计均值和标准差，并在同一横轴上绘制高斯分布曲线。\n"
            "4. 曲线越靠左，说明该方法生成的轨迹整体越接近专家轨迹分布。\n"
        )

    print(f"saved figure: {out_png}")
    print(f"saved summary: {OUTPUT_DIR / 'gaussian_1d_summary.csv'}")


if __name__ == "__main__":
    main()
