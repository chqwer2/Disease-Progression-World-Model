"""
AD-Progression triplet dataloader (Step-1 contrastive AE).

The AD-Progression release ships a FLAT master CSV — one row per timepoint
(`AD-Progression-All.csv`) — plus an `Image/<subject>/<date>/{t1,seg,
mask,energy}.nii.gz` tree.  The Step-1 contrastive trainer, however, consumes
time-ORDERED triplets (`starting`, `followup`, `followup2`) of the SAME patient.

This module builds those triplets on the fly from the flat CSV:

  * group by `subject_id` (already source-prefixed, e.g. <COHORT>_<subject>),
  * sort each patient's visits by `follow_up` (months since baseline),
  * emit consecutive ordered triples (i, i+1, i+2) so every visit is covered
    and a1 < a2 < a3 holds by construction (monotone for the ArcRank loss),
  * for 2-visit patients (no triple possible) emit a duplicate-last triple
    (t1, t2, t2) so BOTH real images still receive the reconstruction loss —
    the rank term skips it because a2 == a3 is not strictly increasing.

Half resolution: the whole brain is resampled to the 1.5 mm canonical grid
(128,144,128) and then DOWNSAMPLED to (64,72,64) (÷8-divisible, latent (8,9,8)).
It is a deterministic whole-brain resize (NOT a random crop) so a patient's three
visits stay voxel-aligned — a prerequisite for measuring latent-trajectory
linearity.

Drop-in replacement for `dataset.oasis_dataset_3D_pair_contrastive.get_brain_dataset`.
"""
import os
import numpy as np
import pandas as pd

import torch
from monai import transforms
from monai.data import Dataset, DataLoader, list_data_collate
from monai.transforms import (
    CopyItemsD, LoadImageD, EnsureChannelFirstD, EnsureTypeD, LambdaD,
    OrientationD, SpacingD, ScaleIntensityD, ResizeWithPadOrCropD, ResizeD,
    RandFlipD, CropForegroundD, RandSpatialCropD, CenterSpatialCropD,
)

from .utils.utils import concat_covariates, get_dataset_from_pd
from .utils import const


# regional columns copied from the STARTING visit so concat_covariates() can
# assemble the cross-attention context (harmless / unused by the Step-1 AE).
_START_COVARS = ["sex", "diagnosis"] + const.CONDITIONING_REGIONS


def _resolve_csv(args) -> str:
    """dataset_csv may be absolute or relative to data_dir."""
    p = args.dataset_csv
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    cand = os.path.join(getattr(args, "data_dir", ""), p)
    return cand if os.path.exists(cand) else p


def _resolve_data_root(data_dir: str) -> str:
    """Find the directory that actually contains the `Image/` tree."""
    for cand in (data_dir, os.path.join(data_dir, "Final")):
        if os.path.isdir(os.path.join(cand, "Image")):
            return cand
    # fall back to data_dir; paths just won't resolve and we'll error loudly
    return data_dir


def build_triplets(csv_path, data_root, strict_time=True, min_tp=3):
    """Flat per-timepoint CSV -> per-patient time-ordered triplet DataFrame.

    Returns a DataFrame with one row per triplet and the columns the Step-1
    transform pipeline expects: `{starting,followup,followup2}_image_path`,
    `{starting,followup,followup2}_age`, `subject_id` (int index), plus the
    conditioning covariates taken from the starting visit.
    """
    df = pd.read_csv(csv_path)

    # map subject string id -> stable integer index (needed by the loss / ids)
    uniq = sorted(df["subject_id"].unique())
    sid2idx = {s: i for i, s in enumerate(uniq)}

    def _abs(p):
        return os.path.join(data_root, str(p))

    rows = []
    for sid, g in df.groupby("subject_id"):
        g = g.sort_values(["follow_up", "capture_time"], kind="mergesort")
        # collapse same-day repeat scans (identical follow_up) -> distinct timepoints,
        # so every triple is STRICTLY increasing in time (clean monotone signal).
        g = g.drop_duplicates(subset="follow_up", keep="first").reset_index(drop=True)
        n = len(g)
        if n < max(2, int(min_tp)):
            continue                        # >=3 distinct timepoints by default
        if n >= 3:
            triples = [(i, i + 1, i + 2) for i in range(n - 2)]
        else:
            # min_tp=2: a 2-visit subject is padded to (v0, v1, v1). Fine for reconstruction and
            # perceptual losses; trajectory / angle / contrastive losses would see a degenerate third point.
            triples = [(0, 1, 1)]

        for (i, j, k) in triples:
            r0, r1, r2 = g.iloc[i], g.iloc[j], g.iloc[k]
            rec = {
                "subject_id": sid2idx[sid],
                "subject_str": sid,
                "source": r0["source"],
                "starting_image_path":  _abs(r0["image_path"]),
                "followup_image_path":  _abs(r1["image_path"]),
                "followup2_image_path": _abs(r2["image_path"]),
                "starting_age":  float(r0["age"]),
                "followup_age":  float(r1["age"]),
                "followup2_age": float(r2["age"]),
                "starting_follow_up": float(r0["follow_up"]),
                "followup_follow_up": float(r1["follow_up"]),
                "followup2_follow_up": float(r2["follow_up"]),
            }
            # covariates from the starting visit (context for later steps)
            rec["sex"] = float(r0.get("sex", 0.5))
            rec["starting_diagnosis"] = float(r0.get("diagnosis", -1))
            for reg in const.CONDITIONING_REGIONS:
                rec[f"starting_{reg}"] = float(r0.get(reg, 0.0))
            # PER-VISIT biomarkers for the disease-anchor loss: hippocampus
            # atrophies and lateral_ventricle enlarges with AD; diagnosis is the label.
            for pfx, rr in (("starting", r0), ("followup", r1), ("followup2", r2)):
                rec[f"{pfx}_hippocampus"] = float(rr.get("hippocampus", 0.0) or 0.0)
                rec[f"{pfx}_lateral_ventricle"] = float(rr.get("lateral_ventricle", 0.0) or 0.0)
                rec[f"{pfx}_diagnosis"] = float(rr.get("diagnosis", -1) if rr.get("diagnosis", -1) == rr.get("diagnosis", -1) else -1)
            rows.append(rec)

    tdf = pd.DataFrame(rows)
    # normalise age to [0,1] if the CSV stored raw years
    for c in ("starting_age", "followup_age", "followup2_age"):
        if (tdf[c] > 1).any():
            tdf[c] = tdf[c] / 100.0
    return tdf


def _split_by_patient(tdf, mode, ratio=0.8):
    """Patient-disjoint train/test split (no timepoint leakage across the split)."""
    pats = np.array(sorted(tdf["subject_str"].unique()))
    # deterministic shuffle by an md5 of the id. The built-in hash() is salted per process for str,
    # so it would change the train/test split between runs.
    import hashlib as _hl
    order = np.argsort([int(_hl.md5(("ap|" + str(p)).encode()).hexdigest()[:8], 16)
                        for p in pats])
    pats = pats[order]
    cut = int(ratio * len(pats))
    if mode == "train":
        keep = set(pats[:cut])
    elif mode == "test":
        keep = set(pats[cut:])
    elif mode == "test_all":
        keep = set(pats)
    else:
        raise ValueError("mode must be train / test / test_all")
    return tdf[tdf["subject_str"].isin(keep)].reset_index(drop=True)


def _image_transforms(image_size, canonical=(128, 144, 128), resolution=1.5, key="image",
                      crop_mode="resize"):
    """Deterministic single-key pipeline. crop_mode='patch' matches the training
    pipeline: NATIVE-res canonical grid, then a CENTERED brain patch (no downsample)
    so a patch-trained AE is evaluated on in-distribution patches."""
    tr = [
        LoadImageD(image_only=True, keys=[key]),
        EnsureChannelFirstD(keys=[key]),
        EnsureTypeD(keys=[key], dtype="float32"),
        ScaleIntensityD(minv=0.0, maxv=1.0, keys=[key]),
        ResizeWithPadOrCropD(spatial_size=canonical, mode="constant",
                             constant_values=0, keys=[key]),
    ]
    if crop_mode == "patch":
        tr += [
            CropForegroundD(keys=[key], source_key=key,
                            select_fn=lambda x: x > 0.02, margin=2, allow_smaller=True),
            CenterSpatialCropD(keys=[key], roi_size=tuple(image_size)),
            ResizeWithPadOrCropD(spatial_size=tuple(image_size), mode="constant",
                                 constant_values=0, keys=[key]),
        ]
    else:
        tr.append(ResizeD(keys=[key], spatial_size=tuple(image_size),
                          mode="trilinear", align_corners=False))
    return transforms.Compose(tr)


def get_visit_dataset(args, mode="test", min_visits=3):
    """One item per DISTINCT visit (for latent-trajectory / linearity probing).

    Returns a loader whose batches carry `image`, `subject_id` (int), `subject_str`,
    and `follow_up`, restricted to patients with >= `min_visits` distinct timepoints.
    Patient-disjoint from the same split logic as the triplet loader.
    """
    df = pd.read_csv(_resolve_csv(args))
    data_root = _resolve_data_root(args.data_dir)
    uniq = sorted(df["subject_id"].unique())
    sid2idx = {s: i for i, s in enumerate(uniq)}

    rows = []
    for sid, g in df.groupby("subject_id"):
        g = g.sort_values(["follow_up", "capture_time"], kind="mergesort")
        g = g.drop_duplicates(subset="follow_up", keep="first").reset_index(drop=True)
        if len(g) < min_visits:
            continue
        for _, r in g.iterrows():
            rows.append({
                "image": os.path.join(data_root, str(r["image_path"])),
                "subject_id": sid2idx[sid],
                "subject_str": sid,
                "follow_up": float(r["follow_up"]),
            })
    vdf = pd.DataFrame(rows)
    vdf = _split_by_patient(vdf.assign(subject_str=vdf["subject_str"]), mode)
    if getattr(args, "DEBUG", False):
        keep = vdf["subject_str"].drop_duplicates().iloc[:8]
        vdf = vdf[vdf["subject_str"].isin(keep)].reset_index(drop=True)

    ds = get_dataset_from_pd(vdf, _image_transforms(args.image_size,
                             crop_mode=getattr(args, "crop_mode", "resize")), None)
    loader = DataLoader(ds, num_workers=args.num_workers,
                        batch_size=getattr(args, "eval_batch_size", args.batch_size),
                        shuffle=False, collate_fn=list_data_collate, pin_memory=False)
    return loader, vdf


def get_brain_dataset(args, mode="train", pair=True, with_image=True, with_latent=False):
    print(f"[ad_progression] building triplets for mode={mode}")

    INPUT_SHAPE_AE = tuple(args.image_size)          # e.g. (64,72,64) half res
    CANONICAL = (128, 144, 128)                       # full-res 1.5mm canonical grid
    RESOLUTION = 1.5

    data_root = _resolve_data_root(args.data_dir)
    tdf = build_triplets(_resolve_csv(args), data_root,
                         min_tp=int(getattr(args, "min_timepoints", 3) or 3))
    tdf = _split_by_patient(tdf, mode)

    if getattr(args, "DEBUG", False):
        tdf = tdf.iloc[:10].reset_index(drop=True)

    print(f"[ad_progression] mode={mode}: {len(tdf)} triplets, "
          f"{tdf['subject_str'].nunique()} patients, data_root={data_root}")

    prefixes = ["starting", "followup", "followup2"]
    imagekeys = prefixes

    trans = [
        CopyItemsD(keys=[i + "_image_path" for i in prefixes], names=prefixes),
        transforms.Lambda(func=concat_covariates) if pair else
            LambdaD(keys=imagekeys, func=lambda x: x),
    ]

    if with_image:
        trans.extend([
            LoadImageD(image_only=True, keys=imagekeys),
            EnsureChannelFirstD(keys=imagekeys),
            EnsureTypeD(keys=imagekeys, dtype="float32"),
            # volumes are already 1.5mm iso + MNI-registered, so no Orientation/Spacing
            # (those trip monai-1.6 meta-dict handling on these headers anyway).
            ScaleIntensityD(minv=0.0, maxv=1.0, keys=imagekeys),
            # canonical full-res grid at NATIVE 1.5mm
            ResizeWithPadOrCropD(spatial_size=CANONICAL, mode="constant",
                                 constant_values=0, keys=imagekeys),
        ])
        if getattr(args, "crop_mode", "resize") == "patch":
            # NATIVE-resolution BRAIN PATCH (matches MAISI's 48^3 sliding-window ROI):
            # crop to the brain bounding box (drop background air), then a fixed-size
            # patch that stays inside the brain — RANDOM location in train (AE sees all
            # regions), CENTERED for eval (deterministic). Triplet-consistent: the 3
            # visits share one crop (registered -> same anatomy), preserving progression.
            trans.append(CropForegroundD(keys=imagekeys, source_key="starting",
                                         select_fn=lambda x: x > 0.02, margin=2,
                                         allow_smaller=True))
            if mode == "train":
                trans.append(RandSpatialCropD(keys=imagekeys, roi_size=INPUT_SHAPE_AE,
                                              random_center=True, random_size=False))
            else:
                trans.append(CenterSpatialCropD(keys=imagekeys, roi_size=INPUT_SHAPE_AE))
            # pad back up if the brain bbox was smaller than the patch on some axis
            trans.append(ResizeWithPadOrCropD(spatial_size=INPUT_SHAPE_AE, mode="constant",
                                              constant_values=0, keys=imagekeys))
        else:
            trans.append(ResizeD(keys=imagekeys, spatial_size=INPUT_SHAPE_AE,
                                 mode="trilinear", align_corners=False))
        if mode == "train":
            _ar = float(getattr(args, "aug_rigid", 0.0))
            if _ar > 0:
                # Rigid augmentation shared across the three visits: the dictionary form of
                #   RandAffineD applies one set of random parameters to every key, so the change
                #   field is preserved. Translation and rotation only, no scaling or elastic warps, which would alter volume, and volume change is the signal.
                from monai.transforms import RandAffineD as _RAffD
                _deg = float(getattr(args, "aug_rot_deg", 5.0)) * 3.14159265 / 180.0
                trans.append(_RAffD(keys=imagekeys, prob=0.5,
                                    rotate_range=(_deg, _deg, _deg),
                                    translate_range=(_ar, _ar, _ar),
                                    scale_range=None, padding_mode="zeros",
                                    mode="bilinear", cache_grid=False))
                print("[aug_rigid] shared across three visits: translation +-%.1f voxels, rotation +-%.1f deg"
                      % (_ar, float(getattr(args, "aug_rot_deg", 5.0))), flush=True)
            trans.extend([
                RandFlipD(prob=0.5, spatial_axis=0, keys=imagekeys),   # L/R flip only;
                # (no A/P or S/I flips: they would fight anatomical priors & the
                #  monotone-progression signal is orientation sensitive)
                EnsureTypeD(keys=imagekeys, dtype="float32"),
            ])

    transforms_fn = transforms.Compose(trans)

    # cache_dir may be unwritable or non-existent (the config prefixes it with
    # temp_path); only use it if it can be created, otherwise run uncached.
    cache_dir = getattr(args, "cache_dir", None) or None
    if cache_dir is not None:
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            cache_dir = None
    trainset = get_dataset_from_pd(tdf, transforms_fn, cache_dir)

    loader = DataLoader(
        dataset=trainset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=(mode == "train"),
        persistent_workers=False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        collate_fn=list_data_collate,
        pin_memory=False,
    )
    return loader, trainset
