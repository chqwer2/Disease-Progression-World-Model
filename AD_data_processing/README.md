# AD_data_processing

Cohort preparation for the longitudinal Alzheimer's-disease datasets used in this
repository. It turns a raw imaging release into the **(baseline, follow-up) pair tables**
that every model here consumes.

This is a shared, standalone stage: both [`ICLR 2026 Delta-LFM/`](../ICLR%202026%20Delta-LFM/)
and [`SMC 2025 MambaControl/`](../SMC%202025%20MambaControl/) build on its output, so it
lives at the top level rather than inside either project. It has no dependency on either
model and can be run on its own.

---

## Inputs

| item | source |
|---|---|
| OASIS pairwise image release | <https://iplab.dmi.unict.it/mfs/pairwise_oasis.zip> |
| `OASIS3_patients_condition.txt` | [TADM repository, `configs/`](https://github.com/MattiaLitrico/TADM-Temporally-Aware-Diffusion-Model-for-Neurodegenerative-Progression-on-Brain-MRI/blob/main/configs/OASIS3_patients_condition.txt) |

The image archive supplies the volumes; the condition file supplies the per-patient
diagnoses. ADNI and AIBL are prepared the same way once their per-visit tables are in the
same format — the pairing logic is cohort-agnostic.

Access to ADNI, AIBL and OASIS requires a separate application and data use agreement; see
the [top-level README](../README.md#data-and-model-weights). **No data is included here.**

---

## Usage

```bash
export RAW_DATA_DIR=/path/to/raw/cohort   # must contain OASISv1.0/
export OUT_DIR=files                      # where the CSVs are written
bash run.sh
```

Or invoke the stages individually — each is independent and re-runnable:

| step | script | output |
|---|---|---|
| 0 | `OASIS-standardize.ipynb` | standardized volumes (run once, manually) |
| 1 | `s1_generate_csv.py` | `s1_OASIS3_ready.csv` — one row per visit |
| 2 | `s2_gather_csv.py` | `oasis_train_{A,B,C,D}.csv` — longitudinal pairs |
| 3 | `s3_clean_data.py` | `oasis_train_B_clean.csv` — pairs failing the registration quality check removed |
| 4 | `build_multiprior.py` | adds `prior_*` / `prior2_*` columns for history conditioning |

Table **A** is per-visit; **B** is the baseline/follow-up pair table used for training.
Step 4 is optional and only needed for the history-conditioned variants.

### Configuration

Nothing is hard-coded. Paths come from arguments or the environment:

| variable / flag | used by | meaning |
|---|---|---|
| `RAW_DATA_DIR` | `run.sh` | root of the downloaded cohort |
| `OUT_DIR` | `run.sh` | destination directory for the CSVs (default `files`) |
| `CONDITION_FILE` | `run.sh` → `s1` | patient condition table |
| `WITH_PRIORS=0` | `run.sh` | skip step 4 |
| `--data_dir`, `--condition_file`, `--output_file` | `s1_generate_csv.py` | |
| `--dataset_csv`, `--output_path`, `--force` | `s2_gather_csv.py` | |
| `ORIG_CSV`, `PSNR_WIDE_CSV`, `OUT_CSV` | `s3_clean_data.py` | |
| `DERIVED_DIR` | `build_multiprior.py` | where the pair tables live |

---

## Output format

One row per (baseline, follow-up) pair. The columns the training code requires:

| column | meaning |
|---|---|
| `subject_id` | patient identifier; splits are always made **per subject** |
| `starting_image_path`, `followup_image_path` | paths to the two volumes |
| `starting_age`, `followup_age` | age at each visit |
| `sex` | encoded 0/1 |
| `prior_image_path`, `has_prior`, `prior2_*`, `has_prior2` | optional, added by step 4 |

Visit coverage is irregular: subjects may be missing individual timepoints, and the
pairing step accounts for this rather than assuming a fixed interval.

---

## Other files

| file | purpose |
|---|---|
| `const.py` | SynthSeg label map, used to derive regional volumes |
| `brain_seg_visualization.py` | quick visual check of a segmentation against its volume |

---

## License

Apache License 2.0, as for the rest of the repository — see [LICENSE](../LICENSE).
