# Disease Progression World Model

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![ICLR 2026](https://img.shields.io/badge/ICLR%202026-%CE%94--LFM-8C1B13.svg)](https://openreview.net/forum?id=cuGnuOfQ4U)
[![IEEE SMC 2025](https://img.shields.io/badge/IEEE%20SMC%202025-MambaControl-00629B.svg)](https://ieeexplore.ieee.org/document/11343720/)

Generative **world models for disease progression** from longitudinal medical imaging.

Given a patient's **baseline 3D brain MRI** and a **target time** Δt, the models in this
repository predict what that patient's **follow-up scan** will look like. Conditioning uses
only information available at prediction time — the baseline scan, baseline-derived
covariates, and the query interval — so no future information enters the prediction.

This is a research monorepo: each paper lives in its own self-contained directory with its
own README, environment, and reproduction recipe.

---

## Projects

| Project | Venue | Approach | Paper | Code |
|---|---|---|---|---|
| **Δ-LFM** — Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation | ICLR 2026 | Flow matching over the latent **change** δ = z₁ − z₀, with an ArcRank-ordered latent space | [OpenReview](https://openreview.net/forum?id=cuGnuOfQ4U) | [`ICLR 2026 Delta-LFM/`](ICLR%202026%20Delta-LFM/) |
| **MambaControl** — Anatomy Graph-Enhanced Mamba ControlNet with Fourier Refinement for Diffusion-Based Disease Trajectory Prediction | IEEE SMC 2025 | Mamba latent diffusion steered by a Mamba ControlNet | [IEEE Xplore](https://ieeexplore.ieee.org/document/11343720/) · [DOI](https://doi.org/10.1109/SMC58881.2025.11343720) | [`SMC 2025 MambaControl/`](SMC%202025%20MambaControl/) |

Both operate in the latent space of a [MAISI](https://github.com/Project-MONAI/tutorials/tree/main/generation/maisi)
3D autoencoder and are evaluated with **change-aware metrics** rather than raw PSNR/SSIM —
copying the baseline scan already scores near-optimal PSNR on this task, so image-fidelity
metrics alone cannot show whether a model predicted change.

### Shared components

| Directory | Purpose |
|---|---|
| [`AD_data_processing/`](AD_data_processing/) | **Stage 0 for both projects.** Turns a raw cohort release into the (baseline, follow-up) pair tables every model consumes: per-visit table → longitudinal pairs → quality filtering → optional prior-visit columns. Runs standalone, with no dependency on either model. |

---

## Repository layout

```
AD_data_processing/        Stage 0 — cohort pair tables (shared)
ICLR 2026 Delta-LFM/       Stage 1-4 — autoencoder, latents, flow matching, evaluation
SMC 2025 MambaControl/     Stage 1-5 — autoencoder, latents, diffusion, ControlNet, inference
```

## Getting started

Each project is self-contained. Every project has a **`README.md`** for orientation and a
**`RECIPE.md`** with the exact command sequence that reproduces its reported numbers.

```bash
# Stage 0 — build the cohort pair tables (needed by both projects)
cd AD_data_processing     && less README.md

# ICLR 2026 — Δ-LFM
cd "ICLR 2026 Delta-LFM"   && less RECIPE.md

# IEEE SMC 2025 — MambaControl
cd "SMC 2025 MambaControl" && less RECIPE.md
```

Both require Linux with an NVIDIA GPU and Python ≥ 3.10. MambaControl additionally needs a
CUDA toolchain, since the Mamba kernels are CUDA-only. All paths are configured through
environment variables; nothing machine-specific is hard-coded.

---

## Data and model weights

**No imaging data and no trained checkpoints are distributed in this repository.**

Experiments use the ADNI, AIBL and OASIS cohorts. Each requires its own application and
data use agreement, and none permits redistribution — obtain access directly from the
providers below and follow the preprocessing steps in the project READMEs.

| Cohort | Access |
|---|---|
| ADNI | <https://adni.loni.usc.edu> |
| AIBL | <https://aibl.org.au> |
| OASIS | <https://sites.wustl.edu/oasisbrains/> |

The pretrained MAISI autoencoder is downloaded from Project MONAI on first use.

---

## Citation

If you use this code, please cite the corresponding paper.

```bibtex
@inproceedings{chen2026deltalfm,
  title     = {Learning Patient-Specific Disease Dynamics With Latent Flow Matching
               For Longitudinal Imaging Generation},
  author    = {Chen, Hao and Yin, Rui and Chen, Yifan and Chen, Qi and Li, Chao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=cuGnuOfQ4U}
}

@inproceedings{yang2025mambacontrol,
  title     = {MambaControl: Anatomy Graph-Enhanced Mamba ControlNet with Fourier
               Refinement for Diffusion-Based Disease Trajectory Prediction},
  author    = {Yang, Hao and Tan, Tao and Tan, Shuai and Yang, Weiqin and
               Cai, Kunyan and Chen, Calvin and Sun, Yue},
  booktitle = {2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  pages     = {4423--4428},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/SMC58881.2025.11343720}
}
```

---

## Maintainer

Maintained by **Hao Chen** ([@chqwer2](https://github.com/chqwer2)), University of Cambridge.

- **Questions, bugs, reproduction issues** — please open an
  [issue](https://github.com/chqwer2/Disease-Progression-World-Model/issues); this is the
  fastest route and keeps answers searchable for others.
- **Collaboration or other enquiries** — <hc666@cam.ac.uk>

Full author lists for each paper are given in the project READMEs and the citations above.

---

## Intended use

This is **research code**. The models generate hypothetical follow-up images and are not
validated for clinical use. They must not be used to inform diagnosis, prognosis, or
patient management.

---

## License

Released under the Apache License 2.0 — see [LICENSE](LICENSE), which applies to both
projects. Vendored third-party components (notably MAISI, NVIDIA / Project MONAI) retain
their own notices in the corresponding source files.

## Acknowledgements

The 3D autoencoder builds on **MAISI** (Project MONAI). The Δ-LFM latent progression
scaffolding derives in part from the open-source Brain Latent Progression codebase
(Puglisi et al., MICCAI 2024 / *Medical Image Analysis* 2025). Data were provided by the
ADNI, AIBL and OASIS studies; we thank the participants and investigators of each.
