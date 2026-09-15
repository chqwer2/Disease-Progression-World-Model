import os
from typing import Optional, Union
from monai.data import Dataset, PersistentDataset
from monai.transforms.transform import Transform

import pandas as pd
from . import const
import torch

import torch.nn.functional as F


def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """
    c1 = const.CONDITIONING_VARIABLES[0]
    if c1 not in _dict:
        return _dict

    _dict['context'] = torch.tensor([_dict[c] for c in const.CONDITIONING_VARIABLES]).unsqueeze(0)

    return _dict



def get_dataframe(args, mode):
    dataset_df = pd.read_csv(args.dataset_csv)

    print("length of dataset_df: ", len(dataset_df))

    if mode == 'train':
        train_df = dataset_df[:int(0.8 * len(dataset_df))]
    elif mode == 'test_all':
        train_df = dataset_df
    elif mode == 'test':
        train_df = dataset_df[int(0.8 * len(dataset_df)):]
    else:
        raise ValueError("Invalid mode. Choose 'train', 'test', or 'test_all'.")

    if args.DEBUG:
        train_df = train_df[:10]

    # ----- Norm age -----
    for age_col in ["starting_age", "followup_age"]:
        if age_col in train_df.columns and np.any(train_df[age_col] > 1):
            train_df[age_col] /= 100

    if "sex" not in train_df.columns or train_df["sex"].nunique() <= 1:
        train_df["sex"] = 0.5  # fallback used only when the sex column is missing

    return train_df


def get_dataset_from_pd(df: pd.DataFrame, transforms_fn: Transform, cache_dir: Optional[str]):
    assert cache_dir is None or os.path.exists(cache_dir), 'Invalid cache directory path'
    data = df.to_dict(orient='records')
    cache_dir = None

    return Dataset(data=data, transform=transforms_fn) if cache_dir is None \
        else PersistentDataset(data=data, transform=transforms_fn, cache_dir=cache_dir)





import os
from typing import Optional, Union

import pandas as pd
from monai.data import Dataset, PersistentDataset
from monai.transforms.transform import Transform

import pandas as pd
import torch.distributed as dist
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset, SmartCacheDataset
from monai.config import DtypeLike, KeysCollection
# Load CSV file into a DataFrame
# df = pd.read_csv("your_file.csv")
import nibabel as nib

import torch
from . import const
from torch.utils.data import DataLoader
from monai import transforms
from monai.transforms import MapTransform
import random
import numpy as np
from monai.transforms import Compose, LoadImageD, LambdaD  # LoadNumpyD

import numpy as np

from monai.data import MetaTensor
from monai.transforms import MapTransform

from monai.config import DtypeLike, KeysCollection, NdarrayOrTensor
from monai.data import image_writer
from monai.data.image_reader import ImageReader
from monai.transforms.io.array import LoadImage, SaveImage, WriteFileMapping
from monai.transforms.transform import MapTransform, Transform
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix






# ---------- Reindex Segmentation ------
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()


class ReindexSegmentation(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        reader: type[ImageReader] | str | None = None,
        dtype: DtypeLike = np.float32,
        meta_keys: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = True,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        prune_meta_pattern: str | None = None,
        prune_meta_sep: str = ".",
        allow_missing_keys: bool = False,
        expanduser: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(
            reader,
            image_only,
            dtype,
            ensure_channel_first,
            simple_keys,
            prune_meta_pattern,
            prune_meta_sep,
            expanduser,
            *args,
            **kwargs,
        )
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError(
                f"meta_keys should have the same length as keys, got {len(self.keys)} and {len(self.meta_keys)}."
            )
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

        self.storage = {}
        self.meta = {}
        self.coarse_regions = const.COARSE_REGIONS
        self.code_map       = const.SYNTHSEG_CODEMAP

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def process(self, data):
        new_segm = np.zeros_like(data)
        ori_segm = np.zeros_like(data)
        # print("Raw Processseg data unique:", np.unique(data))

        # print("seg data:", data.shape, "unique:", np.unique(data))  # seg data: torch.Size([1, 64, 64, 64]

        for id, (code, region) in enumerate(self.code_map.items()):
            if region == 'background':
                continue
            new_segm[data == code] =  id + 1
            ori_segm[data == code] =  code

        # Remove batch dimension if needed
        if data.dim() == 4 and data.shape[0] == 1:
            data = data.squeeze(0)  # now (D, H, W)

        # One-hot encode: shape becomes (D, H, W) → (D*H*W,) → (D*H*W, num_classes)
        num_classes = len([r for r in self.code_map.values() if r != 'background']) + 1  # include background as 0
        data_onehot = F.one_hot(data.long(), num_classes=num_classes)  # shape: (D, H, W, num_classes)
        data_onehot = data_onehot.permute(3, 0, 1, 2).contiguous()      # (num_classes, D, H, W)

        # data = new_segm  # continous mapping 0-1
        # onehot 
        data = data_onehot.float()  # convert to float tensor
        # print("one-hot seg data:", data.shape, "unique:", torch.unique(data))

        # [
        return data, ori_segm

    def __call__(self, data, reader: ImageReader | None = None):
        """
        Raises:
            KeyError: When not ``self.overwriting`` and key already exists in ``data``.

        """
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            img = d[key]
            img, ori_segm = self.process(img)
            d[key] = img
            d['true_'+key] = ori_segm
        return d


# ---------- Resize with Aspect Ratio and Pad/Crop ---------
from monai.transforms import MapTransform, Resize, ResizeWithPadOrCrop
class ResizeWithAspectRatioAndPad(MapTransform):
    def __init__(self, keys, target_size, mode='area'):
        super().__init__(keys)
        self.target_size = np.array(target_size)
        self.mode = mode

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            original_size = np.array(img.shape[-len(self.target_size):])

            # Compute scaling factor to fit inside target_size
            scale_factors = self.target_size / original_size
            min_scale = np.min(scale_factors)
            new_size = (original_size * min_scale).astype(int)

            # Resize proportionally
            resizer = Resize(spatial_size=new_size, mode=self.mode)
            img_resized = resizer(img)

            # Pad or crop to exact target size
            padcrop = ResizeWithPadOrCrop(spatial_size=self.target_size)
            img_final = padcrop(img_resized)

            d[key] = img_final
        return d



# ---------- Slice 2D from 3D image ----------

class Random2DSliceD(MapTransform):
    """
    Randomly extracts a 2D slice from a 3D image along a given axis.
    """

    def __init__(self, keys, axis=1):
        super().__init__(keys)
        self.axis = axis

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            # img shape: (C, D, H, W) – assume channel first
            if img.ndim != 4:
                raise ValueError(f"Expected 4D tensor for {key}, got {img.shape}")
            c, d1, d2, d3 = img.shape

            # Pick slice index randomly along chosen axis
            self.axis = np.random.randint(3)

            axis_size = [d1, d2, d3][self.axis]
            slice_idx = random.randint(0, axis_size - 1)


            if self.axis == 0:
                d[key] = img[:, slice_idx, :, :] #.unsqueeze(1)  # shape: (C, 1, H, W)
            elif self.axis == 1:
                d[key] = img[:, :, slice_idx, :] #.unsqueeze(2)
            elif self.axis == 2:
                d[key] = img[:, :, :, slice_idx] #.unsqueeze(3)


        return d



# Not used by now
class Get2DSliceD(MapTransform):

    def __init__(self, keys, axis=1):
        super().__init__(keys)
        self.axis = axis

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]

            # img shape: (C, D, H, W) – assume channel first
            if img.ndim != 4:
                raise ValueError(f"Expected 4D tensor for {key}, got {img.shape}")
            c, d1, d2, d3 = img.shape

            axis_size = [d1, d2, d3][self.axis]
            slice_idx = data["slice_index"]    #random.randint(0, axis_size - 1)


            if self.axis == 0:
                d[key] = img[:, slice_idx, :, :] #.unsqueeze(1)  # shape: (C, 1, H, W)
            elif self.axis == 1:
                d[key] = img[:, :, slice_idx, :] #.unsqueeze(2)
            elif self.axis == 2 or self.axis == -1:
                d[key] = img[:, :, :, slice_idx] #.unsqueeze(3)


        return d