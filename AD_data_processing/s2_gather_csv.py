import os
import argparse

import pandas as pd
import nibabel as nib
from tqdm import tqdm
import const

# Single time point CSV A
def make_csv_A(df):
    """
    Creates CSV A, which adds the normalized region volumes in the existing CSV.
    The regional volumes are measured from SynthSeg segmentation and normalized
    using the training samples as a reference. 
    """ 
    coarse_regions  = const.COARSE_REGIONS
    code_map        = const.SYNTHSEG_CODEMAP

    records = []
    for record in tqdm(df.to_dict(orient='records')):

        record['segm_path'] = os.path.realpath(record['segm_path'])

        segm = nib.load(record['segm_path']).get_fdata().round()
        record['head_size'] = (segm > 0).sum()
        
        for region in coarse_regions: 
            record[region] = 0
        
        for code, region in code_map.items():
            if region == 'background': continue
            coarse_region = region.replace('left_', '').replace('right_', '')
            record[coarse_region] += (segm == code).sum()
        
        records.append(record)
    
    csv_a_df = pd.DataFrame(records)

    # Norm each region using min-max scaling
    for region in coarse_regions:
        # normalize volumes using min-max scaling
        train_values = csv_a_df[:][region]  # [ csv_a_df.split == 'train' ]
        minv, maxv = train_values.min(), train_values.max()
        csv_a_df[region] = (csv_a_df[region] - minv) / (maxv - minv)

    # Norm out

    return csv_a_df


def make_csv_B(df):
    """
    Creates CSV B, which contains all possible pairs (x_a, x_b) such that 
    both scans belong to the same patient and scan x_a is acquired before scan x_b.
    """
    sorting_field = 'months_to_screening' if 'months_to_screening' in df.columns else 'age'

    data = []
    for subject_id in tqdm(df.subject_id.unique()):
        subject_df = df[ df.subject_id == subject_id ].sort_values(sorting_field, ascending=True)
        for i in range(len(subject_df)):
            for j in range(i+1, len(subject_df)):
                s_rec = subject_df.iloc[i]
                e_rec = subject_df.iloc[j]
                record = { 'subject_id': s_rec.subject_id }  # 'split': s_rec.split,  'sex': s_rec.sex
                # Keys:
                remaining_columns = set(df.columns).difference(set(record.keys()))
                for column in remaining_columns:
                    record[f'starting_{column}'] = s_rec[column]
                    record[f'followup_{column}'] = e_rec[column]
                data.append(record)
    return pd.DataFrame(data)


def make_csv_C(df):
    """
    Creates CSV C, which contains all possible pairs (x_a, x_a).
    """
    sorting_field = 'months_to_screening' if 'months_to_screening' in df.columns else 'age'

    data = []
    for subject_id in tqdm(df.subject_id.unique()):
        subject_df = df[ df.subject_id == subject_id ].sort_values(sorting_field, ascending=True)
        for i in range(len(subject_df)):
            # for j in range(i+1, len(subject_df)):
            s_rec = subject_df.iloc[i]
            e_rec = subject_df.iloc[i]

            record = { 'subject_id': s_rec.subject_id }  # 'split': s_rec.split
            remaining_columns = set(df.columns).difference(set(record.keys()))
            for column in remaining_columns:
                record[f'starting_{column}'] = s_rec[column]
                record[f'followup_{column}'] = e_rec[column]
            data.append(record)
    return pd.DataFrame(data)


def make_csv_D(df):
    """
    Creates CSV D, which contains all possible **triplets (x1, x2, x3)** for patients
    with at least 3 scans, ordered by time (age or months_to_screening).
    """
    sorting_field = 'months_to_screening' if 'months_to_screening' in df.columns else 'age'

    data = []
    for subject_id in tqdm(df.subject_id.unique()):
        subject_df = df[df.subject_id == subject_id].sort_values(sorting_field, ascending=True)
        if len(subject_df) < 3:
            continue  # Skip patients with fewer than 3 timepoints

        # Generate all ordered triplets: (i, j, k) with i < j < k
        for i in range(len(subject_df) - 2):
            for j in range(i + 1, len(subject_df) - 1):
                for k in range(j + 1, len(subject_df)):
                    rec1 = subject_df.iloc[i]
                    rec2 = subject_df.iloc[j]
                    rec3 = subject_df.iloc[k]

                    record = {
                        'subject_id': rec1.subject_id,
                        # 'sex': rec1.sex,
                    }

                    for col in df.columns.difference(record.keys()):
                        record[f'starting_{col}']  = rec1[col]
                        record[f'followup_{col}']  = rec2[col]
                        record[f'followup2_{col}'] = rec3[col]

                    data.append(record)

    return pd.DataFrame(data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv',      type=str, required=True)
    parser.add_argument('--output_path',      type=str, required=True)
    parser.add_argument('--DEBUG',            action='store_true')
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    # read the dataset
    df = pd.read_csv(args.dataset_csv)
    if args.DEBUG:
        df = df.sample(10)

    print()
    print('> Creating CSV A\n')
    if os.path.exists(os.path.join(args.output_path, 'oasis_train_A.csv')) and not args.force:
        print('CSV A already exists. Loading it.')
        csv_A = pd.read_csv(os.path.join(args.output_path, 'oasis_train_A.csv'))
    else:
        csv_A = make_csv_A(df)
        # single imaging adds the normalized region volumes in the existing CSV.
        csv_A.to_csv(os.path.join(args.output_path, 'oasis_train_A.csv'), index=False)

    print()
    print('> Creating CSV B\n')
    csv_B = make_csv_B(csv_A)
    # pairs (x_a, x_b)
    csv_B.to_csv(os.path.join(args.output_path, 'oasis_train_B.csv'), index=False)
    with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.max_colwidth', None,
            'display.expand_frame_repr', False
    ):
        print("CSV A :", csv_A.head())
        print("CSV B :", csv_B.head())

    # All Self-pair
    csv_C = make_csv_C(csv_A)
    csv_C.to_csv(os.path.join(args.output_path, 'oasis_train_C.csv'), index=False)

    csv_D = make_csv_D(csv_A)
    csv_D.to_csv(os.path.join(args.output_path, 'oasis_train_D.csv'), index=False)