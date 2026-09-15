import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import shutil
from scipy.ndimage import rotate
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.path as mpath

from skimage.measure import label, regionprops


root = "${RAW_DATA_DIR}/Datasets/medical/Brain/OASIS/processed"
output_root = "./seg_vis"


os.makedirs(output_root, exist_ok=True)



def save_middle_slices(img_data, output_prefix, rotate_90=False):
    """Save middle slices in sagittal, coronal, and axial planes as JPG."""

    # if rotate_90:
    #     img_data = rotate(img_data, 90, axes=(0, 1), reshape=False)

    x_mid = img_data.shape[0] // 2
    y_mid = img_data.shape[1] // 2
    z_mid = img_data.shape[2] // 2

    plt.imsave(output_prefix + "_sagittal.jpg", np.rot90(img_data[x_mid, :, :]), cmap='gray')
    plt.imsave(output_prefix + "_coronal.jpg",  np.rot90(img_data[:, y_mid, :]), cmap='gray')
    plt.imsave(output_prefix + "_axial.jpg",    np.rot90(img_data[:, :, z_mid]), cmap='gray')



# Full label-to-name map (merged and individual parts)
label_name_map = {
    0: 'Background',
    2: 'Left_Cerebral_Cortex',
    3: 'Left_Lateral_Ventricle',
    4: 'Third_Ventricle',
    5: 'Left_Cerebellum_White_Matter',
    6: 'Left_Cerebellum_Cortex',
    7: 'Left_Thalamus',
    8: 'Left_Caudate',
    9: 'Left_Putamen',
    10: 'Left_Pallidum',
    11: 'Brain_Stem',
    12: 'Left_Hippocampus',
    13: 'Left_Amygdala',
    14: 'Left_Hippocampus',   # Alternate label for merging, keep for clarity
    15: 'Left_Amygdala',      # Alternate label for merging
    16: 'CSF',
    17: 'Left_Accumbens',
    18: 'Left_Ventral_DC',
    24: 'CSF_Exterior',
    28: 'Vessel',
    41: 'Right_Cerebral_Cortex',
    42: 'Right_Lateral_Ventricle',
    43: 'Right_Inf_Lat_Ventricle',
    44: 'Right_Cerebellum_White_Matter',
    45: 'Right_Cerebellum_Cortex',
    46: 'Right_Thalamus',
    47: 'Right_Caudate',
    48: 'Right_Putamen',
    49: 'Right_Pallidum',
    50: 'Right_Hippocampus',
    51: 'Right_Amygdala',
    52: 'Right_Accumbens',
    53: 'Right_Ventral_DC',
    58: 'Right_Inf_Lat_Ventricle',
    100: 'Hippocampus',
    101: 'Amygdala',
    102: 'Caudate',
    103: 'Putamen',
    104: 'Pallidum',
    105: 'Accumbens',
    106: 'Cerebellum_Cortex',
    107: 'Cerebral_Cortex',
    108: 'Cerebellum_White_Matter',
    109: 'Lateral_Ventricles',
    110: 'Thalamus',
    111: 'Ventral_DC',
}

# Merge definitions
merge_dict = {
    100: [14, 50],
    101: [15, 51],
    102: [8, 47],
    103: [9, 48],
    104: [10, 49],
    105: [17, 52],
    106: [6, 45],
    107: [2, 41],
    108: [5, 44],
    109: [3, 42],
    110: [7, 46],
    111: [18, 53],
}


def merge_similar_parts(slice_2d):
    """
    Merge left/right and similar structures for SynthSeg output.
    """
    merged_slice = slice_2d.copy()

    for new_label, labels_to_merge in merge_dict.items():
        for label in labels_to_merge:
            merged_slice[merged_slice == label] = new_label

    return merged_slice


def save_seg_numbers_as_svg(seg_data, output_path, area_threshold=100):
    """
    Save merged and unmerged, important, and large regions as SVG with region names.
    """
    os.makedirs(os.path.join(output_path, "seg"), exist_ok=True)

    z_mid = seg_data.shape[2] // 2
    slice_2d = seg_data[:, :, z_mid]

    merged_slice = merge_similar_parts(slice_2d)

    unique_numbers = np.unique(merged_slice)
    unique_numbers = unique_numbers[unique_numbers != 0]

    for number in unique_numbers:
        mask = merged_slice == number

        labeled_mask = label(mask)
        props = regionprops(labeled_mask)
        number = int(number)

        # Determine the region name
        region_name = label_name_map.get(number, None)

        if region_name is None:
            raise ValueError(f"Label {number} has no name in label_name_map. Please add it.")

        for region in props:
            if region.area >= area_threshold:
                region_mask = labeled_mask == region.label

                fig, ax = plt.subplots()
                ax.imshow(region_mask, cmap='gray')
                ax.axis('off')

                file_path = os.path.join(output_path, "seg", f"{region_name}_region_{region.label}.svg")
                fig.savefig(file_path, format='svg', bbox_inches='tight', pad_inches=0)
                plt.close(fig)

DEBUG = True
all_patients = os.listdir(root) if not DEBUG else os.listdir(root)[:3]  # Example patient IDs




# Iterate over patients
for patient in tqdm(all_patients):
    patient_path = os.path.join(root, patient)
    if not os.path.isdir(patient_path):
        continue

    patient_output_dir = os.path.join(output_root, patient)
    os.makedirs(patient_output_dir, exist_ok=True)

    # Iterate over timepoints
    for timepoint in os.listdir(patient_path):
        timepoint_path = os.path.join(patient_path, timepoint)
        if not os.path.isdir(timepoint_path):
            continue

        final_img_path = os.path.join(timepoint_path, f"{timepoint}_anat2_final.nii.gz")
        seg_path       = os.path.join(timepoint_path, f"{timepoint}_anat2_synthseg.nii.gz")

        if not os.path.exists(final_img_path):
            print(f"Missing final image for {patient} {timepoint}, skipping...")
            continue

        try:
            # Read image
            img = nib.load(final_img_path)
            img_data = img.get_fdata()

            output_prefix = os.path.join(patient_output_dir, f"{timepoint}")

            save_middle_slices(img_data, output_prefix)

            # Process segmentation
            if os.path.exists(seg_path):
                seg_img = nib.load(seg_path)
                seg_data = seg_img.get_fdata()

                save_seg_numbers_as_svg(seg_data, patient_output_dir)

        except Exception as e:
            print(f"Error processing {patient} {timepoint}: {e}")
