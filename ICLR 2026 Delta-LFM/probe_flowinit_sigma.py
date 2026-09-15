"""Probe the flow-initialization noise scale (--fm_flowinit_sigma) for a latent dataset.

The value depends on the autoencoder, the latent resolution and the cohort, so it is measured,
not copied. Run this once after Stage 2 with the same data / latent / split flags as Stage 3.

It reports the standard deviation of the latent change delta = z1 - z0 over training pairs:
  raw     std(z1 - z0)                     -> the value used by the recipe
  scaled  std(scale_factor * (z1 - z0))    scale_factor = 1 / std(z) over the first 10 training
                                           latents, exactly as step3_train_flowmatching.py computes it
Pass the printed flag to step3_train_flowmatching.py. eval_fm.py reads it back from the
checkpoint's args.json.

usage:
  python probe_flowinit_sigma.py --config config/default/Step3_3D_FM_multidt15.yaml \
      --data_dir $DATA_DIR --latent_path $WORK_DIR/latents --res_scale 0.82 --norm_mode std \
      --fm_hist_mode prev1+prev2 --split_v3 1 [--probe_pairs 1000]
"""
import sys

import numpy as np
import torch

_n = 1000
if "--probe_pairs" in sys.argv:
    _i = sys.argv.index("--probe_pairs")
    _n = int(sys.argv[_i + 1])
    del sys.argv[_i:_i + 2]

from utils.options import args  # noqa: E402  (parses the remaining flags)
from dataset.oasis_dataset_3D_pair_latent import get_brain_dataset  # noqa: E402

_, ds, _ = get_brain_dataset(args, mode="train", with_seg=False)
z = torch.stack([ds[i]["starting_latent"] for i in range(10)], 0).float()
scale_factor = float(1.0 / torch.std(z))

idx = np.random.RandomState(0).choice(len(ds), min(_n, len(ds)), replace=False)
s1 = s2 = 0.0
cnt = 0
for k in idx:
    item = ds[int(k)]
    d = (item["followup_latent"].double() - item["starting_latent"].double())
    s1 += float(d.sum()); s2 += float((d * d).sum()); cnt += d.numel()
mean = s1 / cnt
raw = (s2 / cnt - mean * mean) ** 0.5
print(f"training pairs used : {len(idx)} of {len(ds)}")
print(f"scale_factor        : {scale_factor:.4f}")
print(f"std(z1 - z0)        : raw {raw:.4f} | scaled {raw * scale_factor:.4f}")
print(f"--fm_flowinit_sigma {raw:.3f}")
