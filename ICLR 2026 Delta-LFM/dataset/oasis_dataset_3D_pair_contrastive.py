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

from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, LambdaD, SpacingD, ScaleIntensityD,
    ResizeWithPadOrCropD, RandFlipD, RandRotateD, RandRotate90D,
    RandZoomD, RandAffineD, Rand3DElasticD, RandGridDistortionD,
    EnsureTypeD, OrientationD, RandSpatialCropD
)

from monai.transforms import Compose, LoadImageD, LambdaD  # LoadNumpyD
import numpy as np





def get_dataframe(args, mode):
    dataset_df = pd.read_csv(args.dataset_csv)


    # Add subject_id mapping
    unique_subject_ids = dataset_df['subject_id'].unique()
    subject_id_to_index = {sid: idx for idx, sid in enumerate(unique_subject_ids)}
    dataset_df['subject_id'] = dataset_df['subject_id'].map(subject_id_to_index)

    print("length of dataset_df: ", len(dataset_df))

    ratio = 0.8
    if   mode == 'train':    train_df = dataset_df[:int(ratio * len(dataset_df))]
    elif mode == 'test_all': train_df = dataset_df
    elif mode == 'test':     train_df = dataset_df[int(ratio * len(dataset_df)):]
    else:                    raise ValueError("Invalid mode. Choose 'train', 'test', or 'test_all'.")

    if args.DEBUG:  train_df = train_df[:10]

    return train_df, subject_id_to_index



def get_brain_dataset(args, mode='train', pair=True, with_image=True, with_latent=False):
    print("Set up get_brain_dataset")

    INPUT_SHAPE_AE = args.image_size
    # INPUT_SHAPE_AE = (128, 144, 128)   #  (120, 144, 120)  , original Shape of MRI data: (182, 218, 182)
    RESOLUTION = 1.5
    
    latent_path = args.latent_path
    train_df, subject_id_to_index = get_dataframe(args, mode)


    prefixes = ["starting", "followup", "followup2"]
    # MONAI >= 1.6 Spacingd treats every dictionary key starting with f"{key}_" as a meta dict
    #   and calls .update() on it. The CSV contains starting_age / starting_diagnosis /
    #   starting_hippocampus and other columns with that prefix, which raises. The transform chain therefore uses non-colliding image keys internally and copies them back at the end.
    imagekeys = ["_imS", "_imF", "_imF2"]

    trans = [
        transforms.CopyItemsD(keys=[i + "_image_path" for i in prefixes], names=imagekeys),
        transforms.Lambda(func=concat_covariates) if pair else
            transforms.LambdaD(keys=imagekeys, func=lambda x: x),  # dummy

    ]

    

    if with_image:
        trans.extend([
            LoadImageD(image_only=True, keys=imagekeys),
            EnsureChannelFirstD(keys=imagekeys),

            EnsureTypeD(keys=imagekeys, dtype="float32"),
            LambdaD(keys=imagekeys, func=lambda x: x.clone()),

            SpacingD(pixdim=RESOLUTION, mode="bilinear", keys=imagekeys),
            ScaleIntensityD(minv=0, maxv=1, keys=imagekeys),

            ResizeWithPadOrCropD(spatial_size=(128, 144, 128),  mode='constant', 
                                 constant_values=0, keys=imagekeys),

            RandSpatialCropD(
                keys=imagekeys,
                roi_size=INPUT_SHAPE_AE,   # your target (D,H,W) or (H,W)
                random_center=True,
                random_size=False          # keep roi_size fixed
            )
                        
            # LambdaD(keys=imagekeys, func=lambda x: x.clone().astype('float32')),  # optional: type cast
        ])

        padding_mode = "reflection"

        if mode == 'train':
            trans.extend( [
                RandFlipD(prob=0.5, spatial_axis=0, keys=imagekeys),
                RandFlipD(prob=0.5, spatial_axis=1, keys=imagekeys),
                RandFlipD(prob=0.5, spatial_axis=2, keys=imagekeys),


                # RandAffineD(
                #     keys=imagekeys,
                #     prob=0.5,
                #     rotate_range=(0.15, 0.15, 0.15),
                #     # translate_range=(8, 8, 8),          # voxels
                #     scale_range=(0.1, 0.1, 0.1),        # +/-10%
                #     shear_range=(0.05, 0.05, 0.05),
                #     mode="bilinear",  # {"image": "bilinear", "label": "nearest"} if labelkeys else 
                #     padding_mode=padding_mode,
                # ),
                
                # # elastic (smooth nonrigid; great for anatomy)

                # Rand3DElasticD(
                #     keys=imagekeys,
                #     prob=0.3, sigma_range=(3, 5), magnitude_range=(2, 5),
                #     mode="bilinear",  # {"image": "bilinear", "label": "nearest"} if labelkeys else 
                #     padding_mode=padding_mode,
                #     # as_tensor_output=False
                # ),

                EnsureTypeD(keys=imagekeys, dtype="float32"),

            ])



    if with_latent:
        latent = [i + "_latent" for i in prefixes]
        trans.extend([
            CopyItemsD(keys=prefixes, names=latent),
            LambdaD(keys=latent,
                               func=lambda path: args.latent_path + "/" + path.split('/')[-1].split('.')[0] +
                               f'_{args.task}_{args.dim}D_latent-{args.diffusion}.npz'),

            LambdaD(keys=latent, func=lambda x: np.load(x, allow_pickle=True)['data']),
            EnsureChannelFirstD(keys=latent, channel_dim=0),
            # DivisiblePadD(keys=latent, k=4, mode='constant'),
            LambdaD(keys=latent, func=lambda x: x.clone().astype('float32')),  # optional: type cast
        ])

    # end of the chain: copy the internal image keys back to the external names (step1 reads batch["starting"] and so on)

    trans.append(transforms.CopyItemsD(keys=imagekeys, names=prefixes))


    transforms_fn = transforms.Compose(trans)

    if args.cache_dir is not None:
        os.makedirs(args.cache_dir, exist_ok=True)

    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)

    from monai.data import DataLoader, list_data_collate

    # train_loader = DataLoader(dataset=trainset,
    #                           num_workers=args.num_workers,
    #                           batch_size=args.batch_size,
    #                           shuffle=True if mode == 'train' else False,
    #                           persistent_workers=True,
    #                         #   collate_fn=list_data_collate, 
    #                           pin_memory=True)

    train_loader = DataLoader(dataset=trainset,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True if mode == 'train' else False,
                            persistent_workers=False,
                            prefetch_factor=4 if args.num_workers>0 else None,
                            collate_fn=list_data_collate, 
                            pin_memory=False)

    return train_loader, trainset

