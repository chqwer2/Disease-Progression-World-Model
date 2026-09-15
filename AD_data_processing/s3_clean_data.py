import os
import pandas as pd
import numpy as np

# Inputs
ORIG_CSV = os.environ.get("ORIG_CSV", "files/oasis_train_B.csv")
PSNR_WIDE_CSV = os.environ.get("PSNR_WIDE_CSV", "oasis_psnr_wide.csv")
OUT_CSV = os.environ.get("OUT_CSV", "files/oasis_train_B_clean.csv")

# --- Load ---
orig_df = pd.read_csv(ORIG_CSV)
psnr_wide = pd.read_csv(PSNR_WIDE_CSV, index_col=0)  # subject_id index

print("Original length:", len(orig_df))

# --- Identify PSNR + PATH columns and order by timepoint ---
def time_index(col: str) -> int:
    # col like 't3_psnr' or 't3_path'
    t = col.split("_")[0]  # 't3'
    return int(t[1:])

psnr_cols = sorted([c for c in psnr_wide.columns if c.endswith("_psnr")], key=time_index)
path_cols = sorted([c for c in psnr_wide.columns if c.endswith("_path")], key=time_index)

# --- For each subject, detect timepoints where PSNR increases vs previous valid PSNR ---
bad_paths = set()

for sid, row in psnr_wide.iterrows():
    prev = None
    for t_idx, (ps_col, p_col) in enumerate(zip(psnr_cols, path_cols)):
        ps = row.get(ps_col, np.nan)
        p  = row.get(p_col,  None)

        # baseline t0: just initialize prev if valid, never mark as bad
        if t_idx == 0:
            if pd.notna(ps):
                prev = ps
            continue

        # Skip if PSNR missing OR path missing
        if pd.isna(ps) or p is None or (isinstance(p, float) and np.isnan(p)):
            # do not update prev; just ignore this timepoint
            continue

        if prev is not None and ps > prev:  # increase detected → inconsistent
            bad_paths.add(p)

        # Update prev if current PSNR is valid (even if it increased)
        prev = ps if pd.notna(ps) else prev

# --- Remove only rows whose follow-up path is inconsistent ---
# (We keep all subjects; only specific follow-up rows are dropped.)
mask_bad = orig_df["followup_image_path"].isin(bad_paths)
clean_df = orig_df.loc[~mask_bad].copy()
mask_bad = clean_df["starting_image_path"].isin(bad_paths)
clean_df = clean_df.loc[~mask_bad].copy()

# --- Drop the path columns as requested ---
# cols_to_drop = [c for c in ["starting_image_path", "followup_image_path"] if c in clean_df.columns]
# clean_df.drop(columns=cols_to_drop, inplace=True)

print("Cleaned length:", len(clean_df))

# --- Save ---
clean_df.to_csv(OUT_CSV, index=False)
print(f"Saved cleaned file to: {OUT_CSV}")
