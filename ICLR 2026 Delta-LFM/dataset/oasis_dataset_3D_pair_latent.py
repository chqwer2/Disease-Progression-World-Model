import os
from typing import Optional, Union

import pandas as pd
from monai.data import Dataset, PersistentDataset
from monai.transforms.transform import Transform
import pandas as pd

from torch.utils.data import DataLoader
from monai import transforms
import torch
from .utils.utils import concat_covariates, ReindexSegmentation, ResizeWithAspectRatioAndPad, get_dataframe, get_dataset_from_pd

from monai.transforms import (EnsureChannelFirstD, SpacingD, ResizeWithPadOrCropD, ScaleIntensityD, RandRotateD,
                              RandFlipD,
                              LoadImageD, LambdaD, CopyItemsD)



from monai.transforms import Compose, LoadImageD, LambdaD  # LoadNumpyD
import numpy as np

# Derived-artifact root (population priors, PCA bases, energy maps).
DERIVED_DIR = os.environ.get("DERIVED_DIR",
                             os.path.join(os.environ.get("DATA_DIR", "."), "derived"))






def get_dataframe(args, mode):
    dataset_df = pd.read_csv(args.dataset_csv)


    # Add subject_id mapping
    unique_subject_ids = dataset_df['subject_id'].unique()
    subject_id_to_index = {sid: idx for idx, sid in enumerate(unique_subject_ids)}
    dataset_df['subject_id'] = dataset_df['subject_id'].map(subject_id_to_index)


    for key in ["starting_image_path", "followup_image_path", "followup2_image_path"]:
        if key in dataset_df.columns:
            dataset_df[key.replace("_image_path", "") + "_latent"] = dataset_df[key].apply(
                lambda x: os.path.join(
                    args.latent_path,
                    x.split('/')[-3],
                    x.split('/')[-2],
                    x.split('/')[-1].split('.')[0] + ".npz"
                )
            )

    # print(" dataset_df.columns = ", dataset_df.columns)
    # print("dataset_df[key _latent]: ", dataset_df["starting_latent"][0], dataset_df["starting_image_path"][0])

    # print("length of dataset_df: ", len(dataset_df))

    ratio = 0.8
    if   mode == 'train':    train_df = dataset_df[:int(ratio * len(dataset_df))]
    elif mode == 'test_all': train_df = dataset_df
    elif mode == 'test':     train_df = dataset_df[int(ratio * len(dataset_df)):]
    else:                    raise ValueError("Invalid mode. Choose 'train', 'test', or 'test_all'.")
    if args.DEBUG:  train_df = train_df[:2]

    return train_df, subject_id_to_index



def get_dataframe(args, mode):
    dataset_df = pd.read_csv(args.dataset_csv)


    # Add subject_id mapping
    unique_subject_ids = dataset_df['subject_id'].unique()
    subject_id_to_index = {sid: idx for idx, sid in enumerate(unique_subject_ids)}
    dataset_df['subject_id'] = dataset_df['subject_id'].map(subject_id_to_index)


    for key in ["starting_image_path", "followup_image_path", "followup2_image_path"]:
        if key in dataset_df.columns:
            dataset_df[key.replace("_image_path", "") + "_latent"] = dataset_df[key].apply(
                lambda x: os.path.join(
                    args.latent_path,
                    x.split('/')[-3],
                    x.split('/')[-2],
                    x.split('/')[-1].split('.')[0] + ".npz"
                )
            )

    print("dataset_df[key _latent]: ", dataset_df["starting_latent"][0])

    # precomputed latent-grid energy maps (parallel tree to latents); enables *_energy keys
    _energy_path = getattr(args, "energy_path", "")
    if _energy_path:
        for key in ["starting_image_path", "followup_image_path", "followup2_image_path"]:
            if key in dataset_df.columns:
                dataset_df[key.replace("_image_path", "") + "_energy"] = dataset_df[key].apply(
                    lambda x: os.path.join(
                        _energy_path, x.split('/')[-3], x.split('/')[-2],
                        x.split('/')[-1].split('.')[0] + ".npz"))

    _HM = str(getattr(args, "fm_hist_mode", "none")) + "+" + str(getattr(args, "fm_gain_mode", "none"))
    if float(getattr(args, "fm_histw_lambda", 0.0)) > 0: _HM = _HM + "+etraj"   # loss weighting also needs energy
    _HM = _HM + "+" + str(getattr(args, "fm_spade_mode", "none"))   # components of the SPADE path must be loaded too
    if str(getattr(args, "fm_timewarp", "none")) != "none": _HM = _HM + "+etraj"   # time warping needs energy
    _HM = _HM + "+" + str(getattr(args, "fm_xattn_mode", "none"))   # components of the cross-attention path
    _HM = _HM + "+" + str(getattr(args, "fm_cnet_hist", "none"))   # components of the separate ControlNet condition
    if float(getattr(args, "fm_flowinit_ecov", 0.0)) > 0: _HM = _HM + "+etraj"   # energy-shaped covariance needs etraj
    if int(getattr(args, "fm_eimg_xattn", 0)) or float(getattr(args, "fm_vfin_recw", 0.0)) > 0:
        pass
    if (int(getattr(args, "fm_eimg_xattn", 0)) or float(getattr(args, "fm_vfin_recw", 0.0)) > 0
            or float(getattr(args, "fm_energy_wavg", 0.0)) > 0):
        dataset_df["vfin_img"] = dataset_df["starting_image_path"].apply(
            lambda x: os.path.join(os.path.join(DERIVED_DIR, "vfin_past"),
                                   x.split('/')[-3] + "__" + x.split('/')[-2] + ".npz"))
    _pmsrc = str(getattr(args, "fm_postmask_src", "oracle"))
    if _pmsrc == "etrue":
        # image-resolution energy formula (1-gamma + edge prior + percentile rescaling), used for the post-decode mask
        _eb = os.path.join(DERIVED_DIR, "etrue_img")
        dataset_df["etrue"] = dataset_df["starting_image_path"].apply(
            lambda x: os.path.join(_eb, x.split('/')[-3] + "__" + x.split('/')[-2] + ".npz"))
    if _pmsrc in ("ek", "ekind", "arate"): _HM = _HM + "+etraj"      # output-side energy mask
    if _pmsrc in ("egrow", "eunion"): _HM = _HM + "+egrid"           # multi-timepoint grid
    # ek/ekind/ek2 are channel slices of etraj and must trigger the etraj load
    if any(_c in ("ek", "ekind", "ek2") for _c in _HM.replace("+", " ").split()):
        _HM = _HM + "+etraj"
    if float(getattr(args, "fm_ephi_w", 0.0)) > 0:
        _HM = _HM + "+vslope"   # the energy suppression term needs energy
    if float(getattr(args, "fm_lat_ew", 0.0)) > 0:
        _HM = _HM + "+vslope"   # latent-space reweighting needs energy
    if float(getattr(args, "fm_elossw", 0.0)) > 0:
        _HM = _HM + "+" + str(getattr(args, "fm_elossw_src", "vslope"))   # the energy used for loss weighting must be loaded as well
    if "vslope" in _HM:
        # vslope_lat also uses the flat naming <subj>__<date>.npz
        _sb = os.path.join(DERIVED_DIR, "vslope_lat")
        dataset_df["vslope"] = dataset_df["starting_image_path"].apply(
            lambda x: os.path.join(_sb, x.split('/')[-3] + "__" + x.split('/')[-2] + ".npz"))
        print(f"[vslope] energy columns attached for {len(dataset_df)} rows", flush=True)
        _exs = dataset_df["vslope"].apply(os.path.exists)
        if (~_exs).any():
            print(f"[vslope] {int((~_exs).sum())}/{len(_exs)} missing -> dropped", flush=True)
            dataset_df = dataset_df[_exs].reset_index(drop=True)
    if "vfin" in _HM:
        # vfin_lat uses the flat naming <subj>__<date>.npz, unlike the three-level directories of the other components
        _vb = os.path.join(DERIVED_DIR, "vfin_lat")
        dataset_df["vfin"] = dataset_df["starting_image_path"].apply(
            lambda x: os.path.join(_vb, x.split('/')[-3] + "__" + x.split('/')[-2] + ".npz"))
        _exv = dataset_df["vfin"].apply(os.path.exists)
        if (~_exv).any():
            print(f"[vfin] {int((~_exv).sum())}/{len(_exv)} missing -> dropped", flush=True)
            dataset_df = dataset_df[_exv].reset_index(drop=True)
    # components of both injection paths must be loaded (concat path fm_hist_mode, spatial-modulation path fm_gain_mode)
    for _nm, _dir in (("trajc", "trajc_0875"), ("etraj", "etraj_0875"), ("ietraj", "ietraj_0875"), ("pdrop", "pdrop_0875"), ("egrid", "egrid_0875")):
        if _nm in _HM:
            _bp = os.path.join(DERIVED_DIR, _dir)
            dataset_df[_nm] = dataset_df["starting_image_path"].apply(
                lambda x: os.path.join(_bp, x.split('/')[-3], x.split('/')[-2], x.split('/')[-1].split('.')[0] + ".npz"))
            _ex2 = dataset_df[_nm].apply(os.path.exists)
            if (~_ex2).any():
                print(f"[{_nm}] {int((~_ex2).sum())}/{len(_ex2)} missing -> dropped", flush=True)
                dataset_df = dataset_df[_ex2].reset_index(drop=True)
    if ("traj" in _HM and "trajc" not in _HM) or _HM.startswith("all"):
        _tfp = os.path.join(DERIVED_DIR, "traj_fit_0875")
        dataset_df["traj_fit"] = dataset_df["starting_image_path"].apply(
            lambda x: os.path.join(_tfp, x.split('/')[-3], x.split('/')[-2], x.split('/')[-1].split('.')[0] + ".npz"))
        _ext = dataset_df["traj_fit"].apply(os.path.exists)
        if (~_ext).any():
            print(f"[traj_fit] {int((~_ext).sum())}/{len(_ext)} missing -> dropped", flush=True)
            dataset_df = dataset_df[_ext].reset_index(drop=True)
    if ("first" in _HM or _HM.startswith("base")) and "first_image_path" in dataset_df.columns:
        dataset_df["first_latent"] = dataset_df["first_image_path"].apply(
            lambda x: os.path.join(args.latent_path, x.split('/')[-3], x.split('/')[-2], x.split('/')[-1].split('.')[0] + ".npz"))
        _exf = dataset_df["first_latent"].apply(os.path.exists)
        if (~_exf).any():
            dataset_df.loc[~_exf, "first_latent"] = dataset_df.loc[~_exf, "starting_image_path"].apply(
                lambda x: os.path.join(args.latent_path, x.split('/')[-3], x.split('/')[-2], x.split('/')[-1].split('.')[0] + ".npz"))
            dataset_df.loc[~_exf, "has_first"] = 0
    if "energy" in _HM or int(getattr(args, "fm_hist_concat", 0)) >= 2:
        _ecp = os.path.join(DERIVED_DIR, "energy_cum_0875")
        dataset_df["cum_energy"] = dataset_df["starting_image_path"].apply(
            lambda x: os.path.join(_ecp, x.split('/')[-3], x.split('/')[-2], x.split('/')[-1].split('.')[0] + ".npz"))
        _exc = dataset_df["cum_energy"].apply(os.path.exists)
        if (~_exc).any():
            print(f"[cum_energy] {int((~_exc).sum())}/{len(_exc)} missing -> dropped", flush=True)
            dataset_df = dataset_df[_exc].reset_index(drop=True)

    if str(getattr(args, "fm_cnet_cond", "scan")) == "hist13":
        dataset_df["hist_cond"] = dataset_df["starting_image_path"].apply(
            lambda x: os.path.join(os.path.join(DERIVED_DIR, "history_cond_0875"),
                x.split('/')[-3], x.split('/')[-2], x.split('/')[-1].split('.')[0] + ".npz"))

    if (getattr(args, "fm_history", 0) or getattr(args, "fm_controlnet", 0) or float(getattr(args, "fm_comp_w", 0.0)) > 0 or int(getattr(args, "fm_hist_concat", 0)) or str(getattr(args, "fm_hist_mode", "none")) != "none" or str(getattr(args, "fm_gain_mode", "none")) != "none" or str(getattr(args, "fm_flowinit", "none")) != "none" or str(getattr(args, "fm_spade_mode", "none")) != "none" or str(getattr(args, "fm_xattn_mode", "none")) != "none") and "prior_image_path" in dataset_df.columns:
        _pp = dataset_df["prior_image_path"].fillna("").astype(str)
        _bad = _pp.str.len() < 3   # empty prior -> fall back to starting (placeholder; gated by has_prior=0)
        dataset_df.loc[_bad, "prior_image_path"] = dataset_df.loc[_bad, "starting_image_path"]
        dataset_df["prior_latent"] = dataset_df["prior_image_path"].apply(
            lambda x: os.path.join(args.latent_path, x.split('/')[-3], x.split('/')[-2], x.split('/')[-1].split('.')[0] + ".npz"))
        if "has_prior" not in dataset_df.columns:
            dataset_df["has_prior"] = (~_bad).astype(int)
        # missing latent file -> fall back to starting and clear has_prior, otherwise the loader raises
        _ex1 = dataset_df["prior_latent"].apply(os.path.exists)
        if (~_ex1).any():
            print(f"[prior] {int((~_ex1).sum())}/{len(_ex1)} prior latents missing -> fallback to starting", flush=True)
            dataset_df.loc[~_ex1, "prior_latent"] = dataset_df.loc[~_ex1, "starting_image_path"].apply(
                lambda x: os.path.join(args.latent_path, x.split('/')[-3], x.split('/')[-2], x.split('/')[-1].split('.')[0] + ".npz"))
            dataset_df.loc[~_ex1, "has_prior"] = 0
        # ---- multi-timepoint injection: the second history scan ----
        if "prior2_image_path" in dataset_df.columns:
            _p2 = dataset_df["prior2_image_path"].fillna("").astype(str)
            _b2 = _p2.str.len() < 3
            dataset_df.loc[_b2, "prior2_image_path"] = dataset_df.loc[_b2, "prior_image_path"]
            def _l2(x):
                p = os.path.join(args.latent_path, x.split('/')[-3], x.split('/')[-2], x.split('/')[-1].split('.')[0] + ".npz")
                return p
            dataset_df["prior2_latent"] = dataset_df["prior2_image_path"].apply(_l2)
            # missing latent file -> fall back to prior1 and clear has_prior2
            _ex = dataset_df["prior2_latent"].apply(os.path.exists)
            dataset_df.loc[~_ex, "prior2_latent"] = dataset_df.loc[~_ex, "prior_latent"]
            _ex2b = dataset_df["prior2_latent"].apply(os.path.exists)
            dataset_df.loc[~_ex2b, "prior2_latent"] = dataset_df.loc[~_ex2b, "prior_latent"]
            dataset_df.loc[~_ex2b, "has_prior2"] = 0
            if "has_prior2" not in dataset_df.columns:
                dataset_df["has_prior2"] = 1
            dataset_df.loc[~_ex, "has_prior2"] = 0
            dataset_df.loc[_b2, "has_prior2"] = 0

    print("length of dataset_df: ", len(dataset_df))


    # for age_col in ["starting_age", "followup_age"]:
    #     if age_col in dataset_df.columns and np.any(dataset_df[age_col] > 1):
    #         dataset_df[age_col] /= 100

    if "sex" not in dataset_df.columns or dataset_df["sex"].nunique() <= 1:
        dataset_df["sex"] = 0.5  # fallback used only when the sex column is missing

    ratio = 0.8
    if int(getattr(args, 'split_v3', 0)):
        # split each cohort 8:2 by patient, so all three cohorts have an in-distribution test partition
        import hashlib as _hl3
        _sid3 = dataset_df['starting_image_path'].map(lambda x: str(x).split('/')[-3])

        def _coh3(s):
            u = str(s).upper()
            return 'ADNI' if 'ADNI' in u else ('AIBL' if 'AIBL' in u else 'OASIS')

        _c3 = _sid3.map(_coh3)
        # hash-stratify within each cohort independently so every cohort is split 80/20; a global 80/20 would unbalance the smaller cohorts
        _hv3 = _sid3.map(lambda s: int(_hl3.md5(('v3|' + str(s)).encode()).hexdigest()[:8], 16) % 1000)
        _istr3 = _hv3 < 800
        _p3 = str(getattr(args, 'test_part', 'all')).lower()
        if mode == 'train':
            train_df = dataset_df[_istr3]
        elif mode == 'test_all':
            train_df = dataset_df
        elif mode == 'test':
            _m = ~_istr3
            if _p3 in ('adni', 'aibl', 'oasis'):
                _m = _m & (_c3 == _p3.upper())
            train_df = dataset_df[_m]
        else:
            raise ValueError("Invalid mode.")
        train_df = train_df.reset_index(drop=True)
        _s3 = train_df['starting_image_path'].map(lambda x: str(x).split('/')[-3])
        from collections import Counter as _Ct3
        print("[split_v3] mode=%s part=%s -> %d rows / %d patients %s"
              % (mode, _p3 if mode == 'test' else '-', len(train_df), _s3.nunique(),
                 dict(_Ct3(_s3.map(_coh3)))), flush=True)
    elif int(getattr(args, 'split_v2', 0)):
        # patient-disjoint split, holding out ADNI for the in-distribution test
        import hashlib as _hl
        _sid = dataset_df['starting_image_path'].map(lambda x: str(x).split('/')[-3])
        _isadni = _sid.map(lambda s: 'ADNI' in str(s).upper())
        # md5 for a stable hash: the built-in hash() differs per process and would change the split on every run
        _hv = _sid.map(lambda s: int(_hl.md5(str(s).encode()).hexdigest()[:8], 16) % 1000)
        _part = str(getattr(args, 'test_part', 'adni'))
        if   mode == 'train':     train_df = dataset_df[_isadni & (_hv < 800)]
        elif mode == 'test_all':  train_df = dataset_df
        elif mode == 'test':
            if _part == 'cross':  train_df = dataset_df[~_isadni]           # AIBL + OASIS
            elif _part == 'both': train_df = dataset_df[(~_isadni) | (_isadni & (_hv >= 800))]
            else:                 train_df = dataset_df[_isadni & (_hv >= 800)]   # hold out ADNI
        else: raise ValueError("Invalid mode.")
        train_df = train_df.reset_index(drop=True)
        print("[split_v2] mode=%s part=%s -> %d rows / %d patients"
              % (mode, _part if mode == 'test' else '-', len(train_df),
                 train_df['starting_image_path'].map(lambda x: str(x).split('/')[-3]).nunique()), flush=True)
    elif mode == 'train':    train_df = dataset_df[:int(ratio * len(dataset_df))]
    elif mode == 'test_all': train_df = dataset_df
    elif mode == 'test':     train_df = dataset_df[int(ratio * len(dataset_df)):]
    else:                    raise ValueError("Invalid mode. Choose 'train', 'test', or 'test_all'.")
    if mode == 'test' and int(getattr(args, 'eval_strat', 0)):
        # The rows are grouped by cohort, so splitting the test set by position puts one
        # cohort entirely before the other. Interleaving by within-cohort relative
        # position makes the first N cases cohort-proportional for any N.
        _c = train_df['starting_image_path'].map(lambda x: x.split('/')[-3].split('_')[0])
        _r = train_df.groupby(_c).cumcount()
        _n = _c.map(_c.value_counts())
        _kk = (_r + 0.5) / _n
        if int(getattr(args, 'eval_strat', 0)) >= 2:
            # strat=2 adds a subject-level rotation: the within-cohort order is grouped by
            # subject, so interleaving by cohort alone makes consecutive pairs come from the
            # same person. Ordering first by each pair's index within its subject makes the
            # first N cases come from approximately N distinct subjects.
            _sub = train_df['starting_image_path'].map(lambda x: x.split('/')[-3])
            _rr = train_df.groupby(_sub).cumcount()
            train_df = train_df.assign(_j=_rr, _k=_kk).sort_values(['_j', '_k'], kind='mergesort')
            train_df = train_df.drop(columns=['_j', '_k']).reset_index(drop=True)
        else:
            train_df = train_df.assign(_k=_kk).sort_values('_k', kind='mergesort').drop(columns=['_k']).reset_index(drop=True)
        import collections as _cl
        _h=[q.split('/')[-3].split('_')[0] for q in train_df['starting_image_path'][:60]]
        _hs = len(set(q.split('/')[-3] for q in train_df['starting_image_path'][:60]))
        print('[eval_strat=%d] cohort-proportional interleaving%s; first 60 cases are %s, covering %d subjects'
              % (int(getattr(args, 'eval_strat', 0)),
                 ' + subject rotation' if int(getattr(args, 'eval_strat', 0)) >= 2 else '',
                 str(dict(_cl.Counter(_h))), _hs), flush=True)


    # ---- restrict the training cohorts only (--train_cohort); the test split is unchanged ----
    # Unlike --split_v2, this leaves the test split untouched, so the training cohort is the only variable.
    _tc = str(getattr(args, 'train_cohort', '') or '').lower()
    if mode == 'train' and _tc and _tc != 'all':
        _keep = [c.strip().upper() for c in _tc.split('+')]
        _c = train_df['starting_image_path'].map(
            lambda x: 'ADNI' if 'ADNI' in x.upper() else ('AIBL' if 'AIBL' in x.upper() else 'OASIS'))
        _n0 = len(train_df)
        train_df = train_df[_c.isin(_keep)].reset_index(drop=True)
        print('[train_cohort=%s] training rows %d -> %d (%d patients)'
              % (_tc, _n0, len(train_df),
                 train_df['starting_image_path'].map(lambda x: x.split('/')[-3]).nunique()), flush=True)

    # ---- cap on pairs per patient (--fm_pair_cap), to limit memorization ----
    # Pairs per patient are heavily skewed: a few frequently scanned patients contribute many
    # pairs each, so the model sees the same brain repeatedly. The cap keeps the first N pairs
    # in the original within-subject order, which is deterministic and introduces no seed
    # dependence. Applies to the train split only.
    _cap = int(getattr(args, 'fm_pair_cap', 0))
    if mode == 'train' and _cap > 0:
        _s = train_df['starting_image_path'].map(lambda x: x.split('/')[-3])
        _n0 = len(train_df)
        train_df = train_df[train_df.groupby(_s).cumcount() < _cap].reset_index(drop=True)
        print('[pair_cap=%d] training rows %d -> %d (%d patients)'
              % (_cap, _n0, len(train_df),
                 train_df['starting_image_path'].map(lambda x: x.split('/')[-3]).nunique()), flush=True)

    if args.DEBUG:  train_df = train_df[:10]

    
    return train_df, subject_id_to_index


def get_brain_dataset(args, mode='train', pair=True, with_image=False, with_latent=True, with_seg=True):
    print("Set up get_brain_dataset")

    INPUT_SHAPE_AE = args.image_size
    # INPUT_SHAPE_AE = (128, 144, 128)   #  (120, 144, 120)  , original Shape of MRI data: (182, 218, 182)
    RESOLUTION = 1.5

    latent_path = args.latent_path


    train_df, subject_id_to_index = get_dataframe(args, mode)

    prefixes = ["starting", "followup" ]  # followup2
    imagekeys = prefixes
    maskkeys = [i + "_seg" for i in prefixes] if with_seg else None


    trans = [
        transforms.CopyItemsD(keys=[i + "_image_path" for i in prefixes], names=prefixes),
        transforms.Lambda(func=concat_covariates) if pair else
            transforms.LambdaD(keys=imagekeys, func=lambda x: x),  # dummy

    ]
    if with_seg:
        
        trans.extend([
            transforms.CopyItemsD(keys=[i + "m_path" for i in maskkeys], names=maskkeys),
            LoadImageD(image_only=True, keys=maskkeys),
            EnsureChannelFirstD(keys=maskkeys),
            LambdaD(keys=maskkeys, func=lambda x: x.clone()),
            SpacingD(pixdim=RESOLUTION, mode="nearest", keys=maskkeys),
            ScaleIntensityD(minv=0, maxv=1, keys=maskkeys),
            ResizeWithPadOrCropD(spatial_size=INPUT_SHAPE_AE,  mode='constant', constant_values=0, keys=maskkeys),
            LambdaD(keys=maskkeys, func=lambda x: x.clone().astype('float32')),  # optional: type cast
            # [33, 64, 64, 64]
            ReindexSegmentation(keys=maskkeys),  # 
        ])


    if with_image:
        trans.extend([
            LoadImageD(image_only=True, keys=imagekeys),
            EnsureChannelFirstD(keys=imagekeys),
            LambdaD(keys=imagekeys, func=lambda x: x.clone()),
            SpacingD(pixdim=RESOLUTION, mode="bilinear", keys=imagekeys),
            ScaleIntensityD(minv=0, maxv=1, keys=imagekeys),

            ResizeWithPadOrCropD(spatial_size=INPUT_SHAPE_AE,  mode='constant', constant_values=0, keys=imagekeys),
           
            LambdaD(keys=imagekeys, func=lambda x: x.clone().astype('float32')),  # optional: type cast
        ])

        if mode == 'train':
            trans.extend( [
                RandFlipD(prob=0.5, spatial_axis=0, keys=imagekeys),
                RandRotateD(range_x=0.1, prob=0.5, keys=imagekeys)
            ])



    if with_latent:
        latent = [i + "_latent" for i in prefixes]
        if (getattr(args, "fm_history", 0) or getattr(args, "fm_controlnet", 0) or float(getattr(args, "fm_comp_w", 0.0)) > 0 or int(getattr(args, "fm_hist_concat", 0)) or str(getattr(args, "fm_hist_mode", "none")) != "none" or str(getattr(args, "fm_gain_mode", "none")) != "none" or str(getattr(args, "fm_flowinit", "none")) != "none" or str(getattr(args, "fm_spade_mode", "none")) != "none" or str(getattr(args, "fm_xattn_mode", "none")) != "none") and "prior_latent" in train_df.columns:
            latent = latent + ["prior_latent"]
            if "prior2_latent" in train_df.columns:
                latent = latent + ["prior2_latent"]
        if "cum_energy" in train_df.columns:
            latent = latent + ["cum_energy"]
        if "first_latent" in train_df.columns:
            latent = latent + ["first_latent"]
        if "traj_fit" in train_df.columns:
            latent = latent + ["traj_fit"]
        for _nm in ("trajc", "etraj", "ietraj", "pdrop", "egrid", "vfin", "vslope"):
            if _nm in train_df.columns:
                latent = latent + [_nm]
        trans.extend([
            # CopyItemsD(keys=prefixes, names=latent),
            # LambdaD(keys=latent,
            #                    func=lambda path: 
            #                    args.latent_path + "/" + path.split('/')[-1].split('.')[0] +
            #                    f'_{args.task}_{args.dim}D_latent-{args.diffusion}.npz'),

            LambdaD(keys=latent, func=lambda x: np.load(x, allow_pickle=True)['data']),
            EnsureChannelFirstD(keys=latent, channel_dim=0),
            # DivisiblePadD(keys=latent, k=4, mode='constant'),
            LambdaD(keys=latent, func=lambda x: x.clone().astype('float32')),  # optional: type cast
        ])

    if getattr(args, "energy_path", "") and all((i + "_energy") in train_df.columns for i in prefixes):
        energy = [i + "_energy" for i in prefixes]
        trans.extend([
            LambdaD(keys=energy, func=lambda x: np.load(x, allow_pickle=True)['data']),  # (1,28,32,28) in [0,1]
            EnsureChannelFirstD(keys=energy, channel_dim=0),
            LambdaD(keys=energy, func=lambda x: x.clone().astype('float32')),
        ])

    if str(getattr(args, "fm_cnet_cond", "scan")) == "hist13" and "hist_cond" in train_df.columns:
        trans.extend([
            LambdaD(keys=["etrue"], func=lambda x: (np.load(x, allow_pickle=True)['data'].astype('float32') if (isinstance(x,str) and os.path.exists(x)) else np.zeros((1,1,1), dtype='float32')), allow_missing_keys=True),
            LambdaD(keys=["hist_cond"], func=lambda x: (np.load(x, allow_pickle=True)['data'] if os.path.exists(x) else np.zeros((13,28,32,28), dtype='float32'))),
            EnsureChannelFirstD(keys=["hist_cond"], channel_dim=0),
            LambdaD(keys=["hist_cond"], func=lambda x: x.clone().astype('float32')),
        ])

    transforms_fn = transforms.Compose(trans)

    # cache_dir "none"/"" -> plain on-the-fly Dataset (no PersistentDataset disk cache).
    # Latents are small (npz) so on-the-fly loading from the page cache is fast and avoids
    # PersistentDataset worker stalls under heavy I/O contention.
    _cache = None if (not args.cache_dir or str(args.cache_dir).lower() == "none") else args.cache_dir
    if _cache is not None:
        os.makedirs(_cache, exist_ok=True)

    trainset = get_dataset_from_pd(train_df, transforms_fn, _cache)

    _pw = bool(_cache is not None and args.num_workers > 0)  # persistent_workers only with cache+workers
    train_loader = DataLoader(dataset=trainset,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=True if mode == 'train' else False,
                              persistent_workers=_pw,
                              pin_memory=True)

    return train_loader, trainset, subject_id_to_index

