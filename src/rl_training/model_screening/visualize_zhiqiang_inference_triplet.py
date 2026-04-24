#!/usr/bin/env python3
import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image, ImageFile


def setup_fonts():
    candidate_fonts = [
        'Noto Sans CJK SC', 'Noto Sans SC', 'Source Han Sans SC',
        'WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei',
        'PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS',
    ]
    for font_path in [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    ]:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception:
                pass
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name]
            break
    plt.rcParams['axes.unicode_minus'] = False


def load_gt(path: Path) -> np.ndarray:
    pts = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pts.append([float(row['gt_x']), float(row['gt_y'])])
    return np.asarray(pts, dtype=np.float32)


def load_pred(path: Path):
    by_id = {}
    gc_distance_pred = None
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sid = int(row['sample_id'])
            by_id.setdefault(sid, []).append(
                (int(row['step']), float(row['pred_x']), float(row['pred_y']))
            )
            if gc_distance_pred is None:
                gc_distance_pred = float(row['gc_distance_pred'])
    trajs = {}
    for sid, rows in by_id.items():
        rows = sorted(rows, key=lambda x: x[0])
        trajs[sid] = np.asarray([[x, y] for _, x, y in rows], dtype=np.float32)
    return trajs, gc_distance_pred


def score_traj(pred: np.ndarray, gt: np.ndarray):
    diff = pred - gt
    dists = np.linalg.norm(diff, axis=1)
    ade = float(np.mean(dists))
    fde = float(dists[-1])
    score = 1.0 / (ade + 1e-6) + 1.5 / (fde + 1e-6)
    return score, ade, fde


def render_score_plot(case_id: int, pred_path: Path, gt_path: Path, output_path: Path):
    setup_fonts()
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 120})

    gt = load_gt(gt_path)
    trajs, _gc_distance_pred = load_pred(pred_path)
    scores = {}
    stats = {}
    for sid, traj in trajs.items():
        score, ade, fde = score_traj(traj, gt)
        scores[sid] = score
        stats[sid] = (ade, fde)
    best_sid = max(scores, key=scores.get)

    fig, ax = plt.subplots(figsize=(6.8, 6.1))
    cmap = plt.get_cmap('Blues')
    score_vals = np.asarray(list(scores.values()), dtype=np.float32)
    smin, smax = float(score_vals.min()), float(score_vals.max())

    def norm_score(s):
        if smax - smin < 1e-8:
            return 0.5
        return (s - smin) / (smax - smin)

    for sid, traj in trajs.items():
        if sid == best_sid:
            continue
        c = cmap(0.25 + 0.45 * norm_score(scores[sid]))
        ax.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.35, linewidth=1.4)

    best = trajs[best_sid]
    ax.plot(best[:, 0], best[:, 1], color='#D55E00', linewidth=2.8, zorder=5)
    ax.scatter(0.0, 0.0, color='limegreen', s=110, zorder=6)

    ade, fde = stats[best_sid]
    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.grid(True, alpha=0.25)
    ax.axis('equal')

    legend_items = [
        Line2D([0], [0], color=cmap(0.55), lw=1.6, alpha=0.6, label='候选轨迹'),
        Line2D([0], [0], color='#D55E00', lw=2.8, label='最高分轨迹'),
        Line2D([0], [0], marker='o', color='limegreen', linewidth=0, markersize=9, label='起点'),
    ]
    ax.legend(handles=legend_items, loc='lower right', framealpha=0.92)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return best_sid, scores[best_sid], ade, fde


def compose_triplet(obs_path: Path, score_plot_path: Path, goal_path: Path, output_path: Path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    obs = Image.open(obs_path).convert('RGB')
    mid = Image.open(score_plot_path).convert('RGB')
    goal = Image.open(goal_path).convert('RGB')

    def fit_h(im, h):
        w = int(im.width * h / im.height)
        return im.resize((w, h), Image.LANCZOS)

    H = 768
    obs = fit_h(obs, H)
    mid = fit_h(mid, H)
    goal = fit_h(goal, H)
    pad = 18
    canvas = Image.new('RGB', (obs.width + mid.width + goal.width + pad * 2, H), (255, 255, 255))
    canvas.paste(obs, (0, 0))
    canvas.paste(mid, (obs.width + pad, 0))
    canvas.paste(goal, (obs.width + pad + mid.width + pad, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--case-id', type=int, default=5)
    parser.add_argument('--obs-image', default='')
    parser.add_argument('--final-image', default='')
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for case_id in range(8):
        gt = load_gt(root / f'traj_{case_id:04d}_gt.csv')
        trajs, gc = load_pred(root / f'traj_{case_id:04d}_pred.csv')
        best_sid = None
        best_score = -1.0
        best_ade = None
        best_fde = None
        for sid, traj in trajs.items():
            score, ade, fde = score_traj(traj, gt)
            if score > best_score:
                best_sid, best_score, best_ade, best_fde = sid, score, ade, fde
        summary_rows.append({
            'case_id': case_id,
            'best_sample_id': best_sid,
            'best_score': f'{best_score:.6f}',
            'best_ade_m': f'{best_ade:.6f}',
            'best_fde_m': f'{best_fde:.6f}',
            'gc_distance_pred': f'{gc:.6f}',
            'gt_end_x': f'{gt[-1,0]:.6f}',
            'gt_end_y': f'{gt[-1,1]:.6f}',
        })

    with open(out / 'score_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    case_id = int(args.case_id)
    score_plot_path = out / f'case_{case_id:04d}_score_plot.png'
    best_sid, score, ade, fde = render_score_plot(
        case_id=case_id,
        pred_path=root / f'traj_{case_id:04d}_pred.csv',
        gt_path=root / f'traj_{case_id:04d}_gt.csv',
        output_path=score_plot_path,
    )

    obs_path = Path(args.obs_image) if args.obs_image else (root / f'obs_{case_id:04d}.png')
    final_path = Path(args.final_image) if args.final_image else (root / f'goal_{case_id:04d}.png')

    compose_triplet(
        obs_path=obs_path,
        score_plot_path=score_plot_path,
        goal_path=final_path,
        output_path=out / f'case_{case_id:04d}_triplet.png',
    )
    with open(out / f'case_{case_id:04d}_meta.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['case_id', 'best_sample_id', 'score', 'ade_m', 'fde_m', 'obs_image', 'final_image', 'pred_csv', 'gt_csv']
        )
        writer.writeheader()
        writer.writerow({
            'case_id': case_id,
            'best_sample_id': best_sid,
            'score': f'{score:.6f}',
            'ade_m': f'{ade:.6f}',
            'fde_m': f'{fde:.6f}',
            'obs_image': str(obs_path),
            'final_image': str(final_path),
            'pred_csv': str(root / f'traj_{case_id:04d}_pred.csv'),
            'gt_csv': str(root / f'traj_{case_id:04d}_gt.csv'),
        })
    print(f'Generated case {case_id:04d}: best={best_sid}, score={score:.2f}, ADE={ade:.3f}, FDE={fde:.3f}')
    print(f'Outputs saved to {out}')


if __name__ == '__main__':
    main()
