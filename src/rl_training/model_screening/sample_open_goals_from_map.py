#!/usr/bin/env python3
import argparse
import json
import math
import os
import random

import numpy as np
import yaml
from PIL import Image


def load_map(map_yaml_path):
    with open(map_yaml_path, 'r', encoding='utf-8') as f:
        meta = yaml.safe_load(f)
    image_path = meta['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)
    img = np.array(Image.open(image_path).convert('L'))
    return meta, img


def classify_cells(img, occupied_thresh=0.65, free_thresh=0.196, negate=0):
    if negate:
        occ = img.astype(np.float32) / 255.0
    else:
        occ = (255.0 - img.astype(np.float32)) / 255.0
    free = occ < free_thresh
    occupied = occ > occupied_thresh
    unknown = ~(free | occupied)
    return free, occupied, unknown


def clearance_mask(free_mask, resolution, clearance_m):
    clearance_cells = max(1, int(math.ceil(clearance_m / resolution)))
    ys, xs = np.where(~free_mask)
    bad = np.zeros_like(free_mask, dtype=bool)
    h, w = free_mask.shape
    for y, x in zip(ys, xs):
        y0 = max(0, y - clearance_cells)
        y1 = min(h, y + clearance_cells + 1)
        x0 = max(0, x - clearance_cells)
        x1 = min(w, x + clearance_cells + 1)
        bad[y0:y1, x0:x1] = True
    return free_mask & (~bad)


def map_to_world(mx, my, resolution, origin):
    wx = origin[0] + (mx + 0.5) * resolution
    wy = origin[1] + (my + 0.5) * resolution
    return float(wx), float(wy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-yaml', required=True)
    parser.add_argument('--x-min', type=float, required=True)
    parser.add_argument('--x-max', type=float, required=True)
    parser.add_argument('--y-min', type=float, required=True)
    parser.add_argument('--y-max', type=float, required=True)
    parser.add_argument('--clearance', type=float, default=0.8, help='minimum clearance to obstacles in meters')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output-json', type=str, default=None)
    args = parser.parse_args()

    meta, img = load_map(args.map_yaml)
    resolution = float(meta['resolution'])
    origin = meta['origin']
    free, occupied, unknown = classify_cells(
        img,
        occupied_thresh=float(meta.get('occupied_thresh', 0.65)),
        free_thresh=float(meta.get('free_thresh', 0.196)),
        negate=int(meta.get('negate', 0)),
    )
    valid = clearance_mask(free, resolution, args.clearance)

    coords = []
    for my, mx in zip(*np.where(valid)):
        wx, wy = map_to_world(mx, my, resolution, origin)
        if args.x_min <= wx <= args.x_max and args.y_min <= wy <= args.y_max:
            coords.append([wx, wy])

    rng = random.Random(args.seed)
    rng.shuffle(coords)
    samples = coords[:args.num_samples]

    result = {
        'map_yaml': args.map_yaml,
        'bounds': {
            'x_min': args.x_min, 'x_max': args.x_max,
            'y_min': args.y_min, 'y_max': args.y_max,
        },
        'clearance_m': args.clearance,
        'num_candidates': len(coords),
        'samples': samples,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
