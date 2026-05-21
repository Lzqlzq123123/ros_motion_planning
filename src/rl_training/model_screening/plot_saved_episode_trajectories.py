#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from matplotlib.lines import Line2D


DEFAULT_RUN_DIR = Path(
    "/data/lzq/ros_motion_planning/src/rl_training/model_screening/"
    "table31_case_capture_seq_oldcfg_with_traj_rerun_20260501"
)
CHINESE_FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/data/lzq/.local/share/fonts/windows/simhei.ttf"),
    Path("/data/lzq/.local/share/fonts/custom/simhei.ttf"),
]
CHINESE_FONT_PROP = None


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_trajectory(csv_path):
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"step", "ego_x", "ego_y", "adv_x", "adv_y"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "step": int(float(row["step"])),
                    "ego_x": float(row["ego_x"]),
                    "ego_y": float(row["ego_y"]),
                    "adv_x": float(row["adv_x"]),
                    "adv_y": float(row["adv_y"]),
                }
            )
    return rows


def read_metadata(meta_path):
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_plot_style():
    global CHINESE_FONT_PROP
    for font_path in CHINESE_FONT_CANDIDATES:
        if font_path.exists():
            CHINESE_FONT_PROP = font_manager.FontProperties(fname=str(font_path))
            break
    if CHINESE_FONT_PROP is not None:
        plt.rcParams["font.family"] = CHINESE_FONT_PROP.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def chinese_font(size=None):
    if CHINESE_FONT_PROP is None:
        return None
    prop = CHINESE_FONT_PROP.copy()
    if size is not None:
        prop.set_size(size)
    return prop


def lighten(color_rgb, amount=0.72):
    rgb = np.asarray(color_rgb, dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


def make_gradient_cmap(name, base_hex, light_amount=0.72):
    base_rgb = np.asarray(matplotlib.colors.to_rgb(base_hex))
    light_rgb = lighten(base_rgb, light_amount)
    return LinearSegmentedColormap.from_list(name, [light_rgb, tuple(base_rgb)])


def add_gradient_path(ax, xy, base_color, label, linewidth=3.0, zorder=3):
    if len(xy) == 0:
        return None

    if len(xy) == 1:
        return ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=24,
            c=[base_color],
            label=label,
            zorder=zorder,
        )

    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = make_gradient_cmap(f"{label}_time_gradient", base_color)
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=plt.Normalize(0.0, 1.0),
        linewidth=linewidth,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    lc.set_array(np.linspace(0.0, 1.0, len(segments)))
    ax.add_collection(lc)

    # Invisible handle keeps the legend clean while the actual trajectory uses LineCollection.
    ax.plot([], [], color=base_color, linewidth=linewidth, label=label)
    return lc


def add_start_end_markers(ax, xy, color, marker_label_prefix, zorder=5):
    if len(xy) == 0:
        return
    ax.scatter(
        xy[0, 0],
        xy[0, 1],
        s=52,
        c=[lighten(matplotlib.colors.to_rgb(color), 0.55)],
        marker="o",
        edgecolors="white",
        linewidths=0.9,
        label=f"{marker_label_prefix}起点",
        zorder=zorder,
    )
    ax.scatter(
        xy[-1, 0],
        xy[-1, 1],
        s=76,
        c=[color],
        marker="*",
        edgecolors="#333333",
        linewidths=0.7,
        label=f"{marker_label_prefix}终点",
        zorder=zorder + 1,
    )


def outcome_text(meta):
    collided = parse_bool(meta.get("collided"))
    reached_goal = parse_bool(meta.get("reached_goal"))
    timed_out = parse_bool(meta.get("timed_out"))
    if collided:
        return "碰撞"
    if reached_goal:
        return "到达目标"
    if timed_out:
        return "超时"
    return "未完成"


def episode_number(csv_path):
    name = csv_path.parent.name
    if name.startswith("episode_"):
        return name.split("_", 1)[1]
    return name


def draw_episode(ax, csv_path, title_prefix=None, compact=False, add_legend=True, add_time_note=True):
    rows = read_trajectory(csv_path)
    if not rows:
        return None

    meta = read_metadata(csv_path.with_name("trajectory_meta.json"))
    ego_xy = np.asarray([[r["ego_x"], r["ego_y"]] for r in rows], dtype=float)
    adv_xy = np.asarray([[r["adv_x"], r["adv_y"]] for r in rows], dtype=float)

    ego_color = "#1f4e79"
    adv_color = "#b13a2f"

    linewidth = 2.4 if compact else 3.2
    add_gradient_path(ax, ego_xy, ego_color, "自车轨迹", linewidth=linewidth, zorder=3)
    add_gradient_path(ax, adv_xy, adv_color, "对抗车轨迹", linewidth=linewidth, zorder=4)
    add_start_end_markers(ax, ego_xy, ego_color, "自车", zorder=6)
    add_start_end_markers(ax, adv_xy, adv_color, "对抗车", zorder=7)

    goal_x = meta.get("goal_x")
    goal_y = meta.get("goal_y")
    if goal_x is not None and goal_y is not None:
        ax.scatter(
            [float(goal_x)],
            [float(goal_y)],
            s=90,
            marker="X",
            c=["#2b8c3e"],
            edgecolors="white",
            linewidths=1.0,
            label="目标点",
            zorder=8,
        )

    outcome = outcome_text(meta)
    steps = int(meta.get("steps", len(rows)))
    ep_no = episode_number(csv_path)
    if title_prefix:
        title = f"{title_prefix}  回合{ep_no} | {outcome} | {steps}步"
    else:
        title = f"回合{ep_no} | {outcome} | 步数={steps}"
    ax.set_title(title, fontproperties=chinese_font(9.5 if compact else 11))
    ax.set_xlabel("x / m", fontproperties=chinese_font(9 if compact else 10))
    ax.set_ylabel("y / m", fontproperties=chinese_font(9 if compact else 10))
    ax.grid(True, linestyle="--", alpha=0.22)
    if compact:
        ax.set_aspect("equal", adjustable="datalim")
    else:
        ax.set_aspect("equal", adjustable="box")

    all_points = np.vstack([ego_xy, adv_xy])
    if goal_x is not None and goal_y is not None:
        all_points = np.vstack([all_points, [[float(goal_x), float(goal_y)]]])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    pad = max(0.35, 0.08 * max(x_max - x_min, y_max - y_min))
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    if compact:
        ax.tick_params(labelsize=7.5)
    if add_legend:
        handles, labels = ax.get_legend_handles_labels()
        keep = []
        seen = set()
        for handle, label in zip(handles, labels):
            if label not in seen:
                keep.append((handle, label))
                seen.add(label)
        ax.legend(
            [h for h, _ in keep],
            [l for _, l in keep],
            loc="best",
            prop=chinese_font(7.5),
            frameon=True,
            framealpha=0.9,
        )

    if add_time_note:
        ax.text(
            0.02,
            0.02,
            "颜色由浅到深表示时间推进",
            transform=ax.transAxes,
            fontproperties=chinese_font(6.8 if compact else 8),
            color="#555555",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2.5},
        )
    return meta


def plot_episode(csv_path, out_path, dpi=240):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=dpi)
    meta = draw_episode(ax, csv_path, add_legend=True)
    if meta is None:
        plt.close(fig)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def parse_episode_list(value):
    if not value:
        return []
    episodes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(item))
    return [f"episode_{idx:03d}" for idx in episodes]


def evenly_sample(items, count):
    if len(items) <= count:
        return list(items)
    indices = np.linspace(0, len(items) - 1, count)
    selected = []
    used = set()
    for value in indices:
        idx = int(round(float(value)))
        while idx in used and idx + 1 < len(items):
            idx += 1
        while idx in used and idx - 1 >= 0:
            idx -= 1
        used.add(idx)
        selected.append(items[idx])
    return selected


def choose_grid_episodes(episode_csvs, explicit_episodes=None, count=9):
    by_name = {csv_path.parent.name: csv_path for csv_path in episode_csvs}
    if explicit_episodes:
        missing = [name for name in explicit_episodes if name not in by_name]
        if missing:
            raise FileNotFoundError(f"Missing selected episodes: {missing}")
        return [by_name[name] for name in explicit_episodes]

    collided = []
    for csv_path in episode_csvs:
        meta = read_metadata(csv_path.with_name("trajectory_meta.json"))
        if parse_bool(meta.get("collided")):
            collided.append(csv_path)
    candidates = collided if len(collided) >= count else episode_csvs
    return evenly_sample(candidates, count)


def make_legend_handles():
    ego_color = "#1f4e79"
    adv_color = "#b13a2f"
    return [
        Line2D([0], [0], color=ego_color, lw=2.8, label="自车轨迹"),
        Line2D([0], [0], color=adv_color, lw=2.8, label="对抗车轨迹"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c9d4df", markeredgecolor="white", markersize=7, label="起点"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#333333", markeredgecolor="#333333", markersize=9, label="终点"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#2b8c3e", markeredgecolor="white", markersize=8, label="目标点"),
    ]


def plot_grid(episode_csvs, out_png, out_pdf=None, dpi=320):
    set_plot_style()
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.2), dpi=dpi)
    for i, (ax, csv_path) in enumerate(zip(axes.ravel(), episode_csvs), start=1):
        draw_episode(
            ax,
            csv_path,
            title_prefix=f"({i})",
            compact=True,
            add_legend=False,
            add_time_note=False,
        )
        if i not in {1, 4, 7}:
            ax.set_ylabel("")
        if i not in {7, 8, 9}:
            ax.set_xlabel("")

    for ax in axes.ravel()[len(episode_csvs):]:
        ax.axis("off")

    fig.legend(
        handles=make_legend_handles(),
        loc="lower center",
        ncol=5,
        prop=chinese_font(10),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.text(
        0.5,
        0.055,
        "轨迹颜色由浅到深表示时间推进",
        ha="center",
        fontproperties=chinese_font(10),
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.99), h_pad=1.1, w_pad=1.0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    if out_pdf:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot saved per-episode ego/adversarial trajectories with time gradients."
    )
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=DEFAULT_RUN_DIR / "trajectories" / "ours_diffusion",
        help="Directory containing episode_*/trajectory.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_DIR / "trajectory_plots" / "ours_diffusion",
        help="Directory to write trajectory figures.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=DEFAULT_RUN_DIR / "trajectory_plots" / "ours_diffusion_3x3_trajectory_grid.png",
        help="Output path for the 3x3 paper-ready trajectory grid.",
    )
    parser.add_argument(
        "--grid-pdf-output",
        type=Path,
        default=DEFAULT_RUN_DIR / "trajectory_plots" / "ours_diffusion_3x3_trajectory_grid.pdf",
        help="Optional PDF output path for the 3x3 trajectory grid.",
    )
    parser.add_argument(
        "--grid-episodes",
        type=str,
        default=None,
        help="Optional 1-based episode list for the grid, e.g. 1,2,4,6,8,12,16,19,20.",
    )
    parser.add_argument("--no-grid", action="store_true", help="Only write per-episode figures.")
    args = parser.parse_args()

    episode_csvs = sorted(args.trajectory_root.glob("episode_*/trajectory.csv"))
    if not episode_csvs:
        raise FileNotFoundError(f"No trajectory.csv files found under {args.trajectory_root}")

    written = []
    for csv_path in episode_csvs:
        out_path = args.output_dir / f"{csv_path.parent.name}_trajectory.png"
        if plot_episode(csv_path, out_path, dpi=args.dpi):
            written.append(out_path)

    print(f"Wrote {len(written)} trajectory figures to {args.output_dir}")
    for path in written:
        print(path)

    if not args.no_grid:
        explicit_episodes = parse_episode_list(args.grid_episodes)
        grid_csvs = choose_grid_episodes(episode_csvs, explicit_episodes=explicit_episodes, count=9)
        plot_grid(grid_csvs, args.grid_output, args.grid_pdf_output, dpi=max(args.dpi, 320))
        print("Wrote 3x3 trajectory grid:")
        print(args.grid_output)
        print(args.grid_pdf_output)
        print("Grid episodes:", ", ".join(path.parent.name for path in grid_csvs))


if __name__ == "__main__":
    main()
