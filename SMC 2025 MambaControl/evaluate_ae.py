"""Evaluate a MambaControl MAISI-AE checkpoint: 3D PSNR / SSIM / MAE on held-out volumes.

The training script logs only losses (no metrics, no val split), so this is the
formal reconstruction-quality evaluator. Rebuilds the EXACT training transform
(1.5mm -> (120,144,120), intensity 0..1), reconstructs a fixed random sample of
volumes, and writes a JSON + human summary.

Usage (from the repo root):
    python evaluate_ae.py --ckpt <path> [--csv <csv>] [--n 32] [--seed 0] [--out <json>]

Design notes:
  * norm_float16 -> False: MAISI's MaisiGroupNorm3D emits fp16 unconditionally, which
    collides with the fp32 convs that follow. Disabling it gives clean fp32 end-to-end,
    so the evaluator runs on CPU or any GPU arch without autocast.
  * device-adaptive: runs on CPU or CUDA. A CUDA build without kernels for the local GPU
    architecture will fail, so check that the torch wheel matches the device.
  * fixed-seed random sample, not df.tail(): tail() biases toward one cohort/ordering.
"""
import argparse, json, os, random
import numpy as np
import pandas as pd
import torch
from monai import transforms
from monai.data import Dataset
from monai.metrics import PSNRMetric, SSIMMetric
from torch.utils.data import DataLoader
from mambacontrol import const, init_autoencoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", default=os.environ.get("MC_AE_CSV",
                   "data/scans.csv"))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", DEV, "| ckpt:", args.ckpt)

    tf = transforms.Compose([
        transforms.CopyItemsD(keys={"image_path"}, names=["img"]),
        transforms.LoadImageD(image_only=True, keys=["img"]),
        transforms.EnsureChannelFirstD(keys=["img"]),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=["img"]),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode="minimum", keys=["img"]),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=["img"]),
    ])

    df = pd.read_csv(args.csv)
    random.seed(args.seed)
    idx = random.sample(range(len(df)), min(args.n, len(df)))
    rows = df.iloc[idx].to_dict("records")
    ds = Dataset(rows, tf)
    dl = DataLoader(ds, batch_size=2, num_workers=4)

    ae = init_autoencoder(args.ckpt).to(DEV).eval()
    n_fixed = 0
    for m in ae.modules():
        if hasattr(m, "norm_float16"):
            m.norm_float16 = False
            n_fixed += 1
    print("disabled norm_float16 on", n_fixed, "modules -> fp32 end-to-end")

    psnr_m = PSNRMetric(max_val=1.0)
    ssim_m = SSIMMetric(spatial_dims=3, data_range=1.0, win_size=7)

    ps, ss, ma = [], [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["img"].to(DEV).float()
            recon, _, _ = ae(x)
            recon = recon.clamp(0, 1).float()
            for i in range(x.shape[0]):
                xi, ri = x[i:i+1], recon[i:i+1]
                ps.append(float(psnr_m(ri, xi).item()))
                ss.append(float(ssim_m(ri, xi).item()))
                ma.append(float(torch.mean(torch.abs(ri - xi)).item()))

    epoch = None
    b = os.path.basename(args.ckpt)
    if "ep-" in b:
        try: epoch = int(b.split("ep-")[1].split(".")[0])
        except Exception: pass

    res = {
        "ckpt": args.ckpt, "epoch": epoch, "n_volumes": len(ps),
        "resolution_mm": const.RESOLUTION, "shape": list(const.INPUT_SHAPE_AE),
        "psnr_mean": float(np.mean(ps)), "psnr_min": float(np.min(ps)), "psnr_max": float(np.max(ps)),
        "ssim_mean": float(np.mean(ss)), "ssim_min": float(np.min(ss)), "ssim_max": float(np.max(ss)),
        "mae_mean": float(np.mean(ma)),
    }

    print("\n=== AE reconstruction quality (3D, full {}) ===".format(list(const.INPUT_SHAPE_AE)))
    print("epoch      : {}".format(epoch))
    print("N volumes  : {}".format(len(ps)))
    print("PSNR  (dB) : mean {:.2f}   min {:.2f}   max {:.2f}".format(res["psnr_mean"], res["psnr_min"], res["psnr_max"]))
    print("SSIM       : mean {:.4f}  min {:.4f}  max {:.4f}".format(res["ssim_mean"], res["ssim_min"], res["ssim_max"]))
    print("MAE        : mean {:.5f}".format(res["mae_mean"]))

    out = args.out or os.path.join(os.path.dirname(args.ckpt) or ".", "eval_{}.json".format(epoch if epoch is not None else "latest"))
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote:", out)


if __name__ == "__main__":
    main()
