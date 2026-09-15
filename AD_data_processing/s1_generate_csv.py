import os
import re
import pandas as pd
import numpy as np
# Argument parsing
import argparse
parser = argparse.ArgumentParser(description="Process OASIS3 patient data.")
parser.add_argument("--data_dir", type=str, required=True, help="Path to the OASIS data directory")
parser.add_argument("--condition_file", type=str, default="configs/OASIS3_patients_condition.txt",
                    help="Tab-separated OASIS3 patient condition table")
parser.add_argument("--output_file", type=str, default="files/s1_OASIS3_ready.csv",
                    help="Destination CSV")
args = parser.parse_args()



# Paths
input_file  = args.condition_file
data_dir    = args.data_dir
output_file = args.output_file



# Read patient condition file
condition_df = pd.read_csv(input_file, sep="\t")  # Assuming tab-separated file

# Clean patient_id and normalize diagnosis
condition_df["patient_id"] = condition_df["OASISID"].str.replace("OAS", "", regex=False)
condition_df["diagnosis"] = condition_df["Condition"] / 3  # Normalize 0-3 to 0-1

records = []

for patient_id in os.listdir(data_dir):
    patient_path = os.path.join(data_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    patient_id_clean = patient_id.replace("OAS", "")

    # Get rows for this patient
    patient_rows = condition_df[condition_df["patient_id"] == patient_id_clean].copy()
    if patient_rows.empty:
        continue

    # Get age_at_visit at timepoint 0 for this patient
    row_time0 = patient_rows.loc[patient_rows["days_to_visit"].idxmin()]
    age_at_time0 = row_time0["age_at_visit"]
    days_at_time0 = row_time0["days_to_visit"]

    for image_id in os.listdir(patient_path):
        image_path = os.path.join(patient_path, image_id)
        if not os.path.isdir(image_path):
            continue

        # Extract timepoint from image_id
        try:
            image_timepoint = int(image_id)
        except ValueError:
            continue

        # Locate *_final.nii.gz and *_synthseg.nii.gz
        final_img = ""
        synthseg_img = ""

        for f in os.listdir(image_path):
            if f.endswith("_final.nii.gz"):
                final_img = os.path.join(image_path, f)
            elif f.endswith("_synthseg.nii.gz"):
                synthseg_img = os.path.join(image_path, f)

        if not final_img or not synthseg_img:
            continue  # Skip incomplete entries

        # Calculate real age at image timepoint
        age_at_image = age_at_time0 + (image_timepoint - days_at_time0) / 365.25
        normalized_age = age_at_image / 100  # Normalize to 0-1

        # print("patient_rows[days_to_visit]", patient_rows["days_to_visit"], image_timepoint)
        closest_idx = (patient_rows["days_to_visit"] - image_timepoint).abs().idxmin()
        # print("closest_idx =", closest_idx)

        closest_row = patient_rows.loc[closest_idx]
        # single positional indexer is out-of-bounds

        diagnosis = closest_row["diagnosis"]

        record = {
            "subject_id": int(patient_id_clean),
            "image_uid": int(image_id),
            "age": normalized_age,
            "diagnosis": diagnosis,
            "image_path": final_img,
            "segm_path": synthseg_img
        }
        records.append(record)

# Final dataframe
final_df = pd.DataFrame(records)

# Sort by patient_id and image_id
final_df = final_df.sort_values(by=["subject_id", "image_uid"])

# Save to CSV
final_df.to_csv(output_file, index=False)
print(f"Saved {len(final_df)} records to {output_file}")
