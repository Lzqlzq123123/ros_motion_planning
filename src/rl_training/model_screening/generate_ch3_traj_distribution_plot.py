#!/usr/bin/env python3
import csv
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import font_manager
import numpy as np


DATASET_DIR = Path("/data/lzq/datasets/gazebo")
METHOD_BASE = Path(
    "/data/lzq/ros_motion_planning/src/rl_training/model_screening/fixed_goal_m3_m4_random_20260323"
)
SCRIPT_PATH = Path(__file__).resolve()

OUTPUT_DIR = Path(
    "/data/lzq/tjuthesis/figures/paper_data_audit_20260324/"
    "fig_3_distribution_consistency_vs_dataset"
)
FINAL_FIGURE_DIR = Path("/data/lzq/tjuthesis/figures/diffusion")

METHODS = [
    ("dataset", "数据集专家轨迹"),
    ("rule", "规则方法"),
    ("planner", "规划器驱动"),
    ("unconditional", "无视觉条件扩散"),
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
PCA_DIM = 2
ELLIPSE_STD = 2.0


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
    meta_rows = []
    for traj_file in sorted(DATASET_DIR.glob("*/traj_data.pkl")):
        with open(traj_file, "rb") as f:
            obj = pickle.load(f)
        raw = np.asarray(obj["position"], dtype=float)
        proc = align_and_normalize_shape(resample_polyline(raw))
        trajs.append(proc)
        meta_rows.append(
            {
                "group_key": "dataset",
                "group_name": "数据集专家轨迹",
                "source_path": str(traj_file),
                "num_raw_points": len(raw),
                "raw_path_length": float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum()),
            }
        )
    return np.asarray(trajs, dtype=float), meta_rows


def load_method_trajectories(method_key, method_name):
    traj_root = METHOD_BASE / method_key / "trajectories"
    trajs = []
    meta_rows = []
    for traj_file in sorted(traj_root.glob("episode_*/trajectory.csv")):
        raw = []
        with open(traj_file, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                raw.append([float(row["adv_x"]), float(row["adv_y"])])
        raw = np.asarray(raw, dtype=float)
        proc = align_and_normalize_shape(resample_polyline(raw))
        trajs.append(proc)
        meta_rows.append(
            {
                "group_key": method_key,
                "group_name": method_name,
                "source_path": str(traj_file),
                "num_raw_points": len(raw),
                "raw_path_length": float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum()),
            }
        )
    return np.asarray(trajs, dtype=float), meta_rows


def pca_fit_transform(features, n_components=PCA_DIM):
    x = np.asarray(features, dtype=float)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:n_components]
    scores = xc @ components.T
    explained = (s[:n_components] ** 2) / np.sum(s**2)
    return scores, components, mean, explained


def confidence_ellipse(points, color):
    if len(points) < 2:
        return None
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width, height = 2 * ELLIPSE_STD * np.sqrt(np.maximum(eigvals, 1e-12))
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    center = points.mean(axis=0)
    return Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        lw=1.6,
        alpha=0.12,
        zorder=1,
    )


def style_distribution_ellipse(ellipse, method_key):
    if ellipse is None:
        return
    ellipse.set_linewidth(1.4)
    ellipse.set_linestyle("-")
    ellipse.set_alpha(None)
    if method_key == "dataset":
        ellipse.set_facecolor((0.82, 0.82, 0.82, 0.06))
        ellipse.set_edgecolor("#000000")
    elif method_key == "ours":
        ellipse.set_facecolor((0.95, 0.70, 0.68, 0.06))
        ellipse.set_edgecolor("#ff0000")
    else:
        edge = matplotlib.colors.to_rgba(COLORS[method_key], 0.18)
        face = matplotlib.colors.to_rgba(COLORS[method_key], 0.05)
        ellipse.set_facecolor(face)
        ellipse.set_edgecolor(edge)
        return


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ensure_dirs()
    set_plot_style()

    group_data = {}
    meta_rows = []

    dataset_trajs, dataset_meta = load_dataset_trajectories()
    group_data["dataset"] = {
        "name": "数据集专家轨迹",
        "trajs": dataset_trajs,
    }
    meta_rows.extend(dataset_meta)

    for method_key, method_name in METHODS[1:]:
        trajs, method_meta = load_method_trajectories(method_key, method_name)
        group_data[method_key] = {"name": method_name, "trajs": trajs}
        meta_rows.extend(method_meta)

    flat_features = []
    group_offsets = {}
    start = 0
    for method_key, _ in METHODS:
        trajs = group_data[method_key]["trajs"]
        flat = trajs.reshape(len(trajs), -1)
        flat_features.append(flat)
        group_offsets[method_key] = (start, start + len(trajs))
        start += len(trajs)

    all_features = np.vstack(flat_features)
    scores, components, mean, explained = pca_fit_transform(all_features, PCA_DIM)

    pca_rows = []
    summary_rows = []
    profile_rows = []

    dataset_scores = scores[group_offsets["dataset"][0] : group_offsets["dataset"][1]]
    dataset_centroid = dataset_scores.mean(axis=0)
    dataset_profile_y = group_data["dataset"]["trajs"][:, :, 1].mean(axis=0)
    dataset_feature_mean = group_data["dataset"]["trajs"].reshape(len(group_data["dataset"]["trajs"]), -1).mean(axis=0)

    for method_key, method_name in METHODS:
        start_idx, end_idx = group_offsets[method_key]
        group_scores = scores[start_idx:end_idx]
        trajs = group_data[method_key]["trajs"]
        feature_mean = trajs.reshape(len(trajs), -1).mean(axis=0)
        centroid_dist = float(np.linalg.norm(group_scores.mean(axis=0) - dataset_centroid))
        feature_mean_dist = float(np.linalg.norm(feature_mean - dataset_feature_mean))

        for local_idx, point in enumerate(group_scores):
            pca_rows.append(
                {
                    "group_key": method_key,
                    "group_name": method_name,
                    "sample_index": local_idx,
                    "pc1": float(point[0]),
                    "pc2": float(point[1]),
                }
            )

        mean_traj = trajs.mean(axis=0)
        std_traj = trajs.std(axis=0)
        progress = np.linspace(0.0, 1.0, trajs.shape[1])
        y_profile_dist = float(np.linalg.norm(mean_traj[:, 1] - dataset_profile_y))
        endpoint_mean = mean_traj[-1]

        summary_rows.append(
            {
                "group_key": method_key,
                "group_name": method_name,
                "num_trajectories": len(trajs),
                "pc1_pc2_centroid_distance_to_dataset": round(centroid_dist, 6),
                "normalized_feature_mean_distance_to_dataset": round(feature_mean_dist, 6),
                "normalized_y_profile_distance_to_dataset": round(y_profile_dist, 6),
                "mean_endpoint_x": round(float(endpoint_mean[0]), 6),
                "mean_endpoint_y": round(float(endpoint_mean[1]), 6),
                "pc1_mean": round(float(group_scores[:, 0].mean()), 6),
                "pc2_mean": round(float(group_scores[:, 1].mean()), 6),
                "pc1_std": round(float(group_scores[:, 0].std()), 6),
                "pc2_std": round(float(group_scores[:, 1].std()), 6),
            }
        )

        for idx, p in enumerate(progress):
            profile_rows.append(
                {
                    "group_key": method_key,
                    "group_name": method_name,
                    "progress": round(float(p), 6),
                    "mean_x": round(float(mean_traj[idx, 0]), 6),
                    "mean_y": round(float(mean_traj[idx, 1]), 6),
                    "std_x": round(float(std_traj[idx, 0]), 6),
                    "std_y": round(float(std_traj[idx, 1]), 6),
                }
            )

    write_csv(
        OUTPUT_DIR / "trajectory_shape_pca_points.csv",
        ["group_key", "group_name", "sample_index", "pc1", "pc2"],
        pca_rows,
    )
    write_csv(
        OUTPUT_DIR / "trajectory_shape_summary.csv",
        [
            "group_key",
            "group_name",
            "num_trajectories",
            "pc1_pc2_centroid_distance_to_dataset",
            "normalized_feature_mean_distance_to_dataset",
            "normalized_y_profile_distance_to_dataset",
            "mean_endpoint_x",
            "mean_endpoint_y",
            "pc1_mean",
            "pc2_mean",
            "pc1_std",
            "pc2_std",
        ],
        summary_rows,
    )
    pca_table_rows = []
    for row in summary_rows:
        pca_table_rows.append(
            {
                "group_name": row["group_name"],
                "num_trajectories": row["num_trajectories"],
                "pca_centroid_distance_to_dataset": row["pc1_pc2_centroid_distance_to_dataset"],
                "normalized_shape_mean_distance_to_dataset": row["normalized_feature_mean_distance_to_dataset"],
                "normalized_profile_distance_to_dataset": row["normalized_y_profile_distance_to_dataset"],
            }
        )
    write_csv(
        OUTPUT_DIR / "trajectory_distribution_table.csv",
        [
            "group_name",
            "num_trajectories",
            "pca_centroid_distance_to_dataset",
            "normalized_shape_mean_distance_to_dataset",
            "normalized_profile_distance_to_dataset",
        ],
        pca_table_rows,
    )
    write_csv(
        OUTPUT_DIR / "trajectory_shape_profiles.csv",
        ["group_key", "group_name", "progress", "mean_x", "mean_y", "std_x", "std_y"],
        profile_rows,
    )
    write_csv(
        OUTPUT_DIR / "trajectory_shape_source_index.csv",
        ["group_key", "group_name", "source_path", "num_raw_points", "raw_path_length"],
        meta_rows,
    )

    with open(OUTPUT_DIR / "plot_method.txt", "w", encoding="utf-8") as f:
        f.write(
            "轨迹分布对比方法说明\n"
            "1. 数据集来源：/data/lzq/datasets/gazebo/*/traj_data.pkl 中的 position。\n"
            "2. 对抗方法来源：fixed_goal_m3_m4_random_20260323 下 rule/planner/unconditional/ours 的 adv 轨迹。\n"
            "3. 每条轨迹统一重采样为 16 个点。\n"
            "4. 先平移到起点为原点，再按前 3 步方向旋转对齐初始朝向。\n"
            "5. 使用轨迹总长度做归一化，仅比较相对形状分布，避免 goal 距离差异造成长度偏置。\n"
            "6. 第一张图展示各方法相对专家数据集的归一化形状偏移距离，距离越小表示分布越接近专家轨迹；第二张图展示归一化进度下的平均横向偏移曲线。\n"
            f"7. 审计目录中保留了 PCA 二维投影数据，前两维累计解释方差比为：{explained.sum():.4f}。\n"
        )

    # figure 1: 1D distribution-friendly bar chart
    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=240)
    bar_labels = [row["group_name"] for row in summary_rows]
    bar_values = [row["normalized_feature_mean_distance_to_dataset"] for row in summary_rows]
    bar_colors = [COLORS[row["group_key"]] for row in summary_rows]
    y_pos = np.arange(len(bar_labels))
    bars = ax.barh(y_pos, bar_values, color=bar_colors, alpha=0.82, edgecolor="white", linewidth=0.8)
    for i, row in enumerate(summary_rows):
        if row["group_key"] == "dataset":
            bars[i].set_edgecolor("#000000")
            bars[i].set_linewidth(1.3)
        elif row["group_key"] == "ours":
            bars[i].set_edgecolor("#c62828")
            bars[i].set_linewidth(1.3)
    x_max = max(bar_values) * 1.12 if bar_values else 1.0
    for i, v in enumerate(bar_values):
        text_x = min(v + 0.008, x_max - 0.01)
        ax.text(text_x, i, f"{v:.3f}", va="center", ha="left", fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("归一化偏移距离")
    ax.grid(axis="x", alpha=0.18, linestyle="--")
    fig.tight_layout()
    pca_png_path = OUTPUT_DIR / "trajectory_distribution_pca.png"
    pca_pdf_path = OUTPUT_DIR / "trajectory_distribution_pca.pdf"
    fig.savefig(pca_png_path, bbox_inches="tight")
    fig.savefig(pca_pdf_path, bbox_inches="tight")
    plt.close(fig)

    # figure 2: profile comparison
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=240)
    progress = np.linspace(0.0, 1.0, RESAMPLE_POINTS)
    for method_key, method_name in METHODS:
        trajs = group_data[method_key]["trajs"]
        mean_y = trajs[:, :, 1].mean(axis=0)
        std_y = trajs[:, :, 1].std(axis=0)
        color = COLORS[method_key]
        lw = 2.6 if method_key == "dataset" else 2.1
        ax.plot(progress, mean_y, color=color, lw=lw, label=method_name)
        ax.fill_between(progress, mean_y - std_y, mean_y + std_y, color=color, alpha=0.12)
    ax.axhline(0.0, color="#999999", lw=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("归一化轨迹进度")
    ax.set_ylabel("归一化横向偏移")
    ax.set_title("轨迹形状剖面对比")
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    profile_png_path = OUTPUT_DIR / "trajectory_shape_profile.png"
    profile_pdf_path = OUTPUT_DIR / "trajectory_shape_profile.pdf"
    fig.savefig(profile_png_path, bbox_inches="tight")
    fig.savefig(profile_pdf_path, bbox_inches="tight")
    plt.close(fig)

    final_pca_png_path = FINAL_FIGURE_DIR / "归一化轨迹形状分布.png"
    final_pca_pdf_path = FINAL_FIGURE_DIR / "归一化轨迹形状分布.pdf"
    final_profile_png_path = FINAL_FIGURE_DIR / "轨迹形状剖面对比.png"
    final_profile_pdf_path = FINAL_FIGURE_DIR / "轨迹形状剖面对比.pdf"
    final_pca_png_path.write_bytes(pca_png_path.read_bytes())
    final_pca_pdf_path.write_bytes(pca_pdf_path.read_bytes())
    final_profile_png_path.write_bytes(profile_png_path.read_bytes())
    final_profile_pdf_path.write_bytes(profile_pdf_path.read_bytes())

    # keep backward-compatible combined copy by reusing PCA figure as main placeholder removed in paper
    compat_png_path = OUTPUT_DIR / "trajectory_distribution_vs_dataset.png"
    compat_pdf_path = OUTPUT_DIR / "trajectory_distribution_vs_dataset.pdf"
    compat_png_path.write_bytes(pca_png_path.read_bytes())
    compat_pdf_path.write_bytes(pca_pdf_path.read_bytes())

    script_copy = OUTPUT_DIR / SCRIPT_PATH.name
    script_copy.write_text(SCRIPT_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"saved PCA figure: {pca_png_path}")
    print(f"saved profile figure: {profile_png_path}")
    print(f"saved summary: {OUTPUT_DIR / 'trajectory_shape_summary.csv'}")


if __name__ == "__main__":
    main()
