#!/usr/bin/env python3
from pathlib import Path
import random
import yaml
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches, font_manager

MAP_YAML = Path('/data/lzq/ros_motion_planning/src/sim_env/maps/warehouse/warehouse.yaml')
OUT_DIR = Path('/data/lzq/tjuthesis/figures/paper_data_audit_20260324/fig_4_curriculum_stage_scene')
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_RANGE = {'x_min': -6.5, 'x_max': -1.0, 'y_min': -5.0, 'y_max': -1.0}
STAGES = [
    ('阶段 I', 1.5, '#4e79a7'),
    ('阶段 II', 2.5, '#59a14f'),
    ('阶段 III', 3.5, '#f28e2b'),
    ('阶段 IV', 5.0, '#e15759'),
]


def set_plot_style():
    font_candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/data/lzq/.local/share/fonts/windows/simsun.ttc',
    ]
    chosen = None
    for p in font_candidates:
        if Path(p).exists():
            try:
                if hasattr(font_manager.fontManager, 'addfont'):
                    font_manager.fontManager.addfont(p)
            except Exception:
                pass
            chosen = font_manager.FontProperties(fname=p).get_name()
            break
    plt.rcParams['font.family'] = chosen or 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = [chosen or 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def load_map():
    data = yaml.safe_load(MAP_YAML.read_text(encoding='utf-8'))
    img_path = (MAP_YAML.parent / data['image']).resolve()
    img = np.array(Image.open(img_path).convert('L'))
    res = float(data['resolution'])
    origin = data['origin']
    h, w = img.shape
    extent = [origin[0], origin[0] + w * res, origin[1], origin[1] + h * res]
    # flip for map coordinates
    img = np.flipud(img)
    return img, extent


def sample_points(rect, n=25, seed=0):
    rnd = random.Random(seed)
    pts = []
    for _ in range(n):
        x = rnd.uniform(rect['x_min'], rect['x_max'])
        y = rnd.uniform(rect['y_min'], rect['y_max'])
        pts.append((x, y))
    return np.array(pts)


def rect_from_span(span):
    return {
        'x_min': BASE_RANGE['x_min'] - span,
        'x_max': BASE_RANGE['x_max'] + span,
        'y_min': BASE_RANGE['y_min'] - span,
        'y_max': BASE_RANGE['y_max'] + span,
    }


def add_rect(ax, rect, color, label=None, linestyle='-'):
    patch = patches.Rectangle(
        (rect['x_min'], rect['y_min']),
        rect['x_max'] - rect['x_min'],
        rect['y_max'] - rect['y_min'],
        linewidth=2.0,
        edgecolor=color,
        facecolor='none',
        linestyle=linestyle,
        label=label,
    )
    ax.add_patch(patch)


def main():
    set_plot_style()
    img, extent = load_map()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2), dpi=220)
    axes = axes.ravel()

    meta_rows = []
    for idx, (name, span, color) in enumerate(STAGES):
        ax = axes[idx]
        ax.imshow(img, cmap='gray', extent=extent, origin='lower')
        current = rect_from_span(span)
        add_rect(ax, BASE_RANGE, '#222222', label='基础目标区域', linestyle='--')
        add_rect(ax, current, color, label=f'{name}采样区域')
        pts = sample_points(current, n=30, seed=idx + 1)
        ax.scatter(pts[:, 0], pts[:, 1], s=10, c=color, alpha=0.75)
        ax.set_title(f'{name}（课程跨度 S={span:.1f}）')
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_aspect('equal')
        ax.grid(alpha=0.12, linestyle='--')
        if idx == 0:
            ax.legend(frameon=False, fontsize=8, loc='lower right')
        meta_rows.append({
            'stage': name,
            'span': span,
            'x_min': current['x_min'],
            'x_max': current['x_max'],
            'y_min': current['y_min'],
            'y_max': current['y_max'],
        })

    for ax in axes:
        ax.set_xlabel('x / m')
        ax.set_ylabel('y / m')

    fig.tight_layout()
    png = OUT_DIR / 'curriculum_stage_scene_overview.png'
    pdf = OUT_DIR / 'curriculum_stage_scene_overview.pdf'
    fig.savefig(png, bbox_inches='tight')
    try:
        fig.savefig(pdf, bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)

    import csv
    with open(OUT_DIR / 'curriculum_stage_scene_ranges.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)

    (OUT_DIR / Path(__file__).name).write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
    print('saved to', OUT_DIR)

if __name__ == '__main__':
    main()
