#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from unidepth.models import UniDepthV2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Absolute path to input image')
    parser.add_argument('--output', required=True, help='Absolute path to output directory')
    parser.add_argument('--model', default='lpiccinelli/unidepth-v2-vitb14')
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UniDepthV2.from_pretrained(args.model).to(device).eval()

    rgb_np = np.array(Image.open(in_path).convert('RGB'))
    rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).to(device)

    with torch.inference_mode():
        pred = model.infer(rgb)

    D = pred['depth'].detach().float().cpu()
    while D.ndim > 2:
        D = D.squeeze(0)
    D = D.numpy()

    finite = np.isfinite(D)
    scale = float(np.percentile(D[finite], 99)) if finite.any() else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    vis = np.clip(D / scale, 0, 1)

    Image.fromarray((vis * 65535.0).astype(np.uint16), mode='I;16').save(out_dir / 'depth.png')

    cmap = plt.get_cmap('turbo')
    colored = (cmap(vis)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(out_dir / 'depth_colored.png')

    print('Saved:', str(out_dir / 'depth.png'))
    print('Saved:', str(out_dir / 'depth_colored.png'))

if __name__ == '__main__':
    main()
