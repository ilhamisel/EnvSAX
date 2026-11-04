# EnvSAX: Envelope-Based Symbolic Representation of Resting-State EEG in Alzheimer's Disease

Code and results for the manuscript *"Envelope-Based Symbolic Representation of Resting-State EEG in
Alzheimer's Disease: A Fidelity-First Evaluation"* (Z. M. Alçin, İ. Sel; under review).

EnvSAX encodes each channel and frequency band of a resting-state EEG segment with two symbol streams: a
**shape code** (SAX of the z-normalised log Hilbert envelope) and a **scale code** (quantile-quantised log band
power). The study asks how much of the AD–control spectral group-difference profile survives symbolisation
(discriminative fidelity, r_d) and, separately, how symbolic inputs perform in classification.

The work was supported by TÜBİTAK under project 126E843.

## Repository layout

```
experiments/                 main pipeline on OpenNeuro ds004504 (36 AD, 29 HC)
  config.py, common.py, symbolic.py
  01_preprocess.py … 11_stats.py
  run_all.sh, requirements.txt
  outputs/                   result tables (T2–T13), out-of-fold predictions, RESULTS_v2.md
supplementary_analyses/
  shape_scale_rd_bootstrap/  shape/scale ablation (Table 8), bootstrap CI of r_d
  abba_check_rd_bandwise/    ABBA reference-component check (Table 5), band-wise r_d, M1–M1s ΔAUC CI
external_validation/         frozen scale code applied to an independent cohort (Sections 2.5 and 3.5)
  03_PROTOCOL/               analysis protocol (original and amended), frozen encoder + hash
  04_EXPERIMENTS/code/       00_build_frozen_encoder.py … 06_results_table.py
  04_EXPERIMENTS/*/          fidelity, sensitivity and classification results
  05_FIGURES/                E1–E5 (SVG, PDF, PNG)
  06_REPORTS/RESULTS_COMPARISON.csv
external_validation_v2/      participant-disjoint, label-free recalibration in the external cohort
  01_PROTOCOL/               protocol, fold assignment, PROTOCOL_LOCK.json (sha256, locked before results)
  02_EXPERIMENTS/code/       01_make_folds_and_lock.py, 02_oof_calibration.py, 03_figures.py, v2lib.py
  02_EXPERIMENTS/*/          fold codebooks, bootstrap replicates, out-of-fold profiles, leakage check
  03_RESULTS/                fidelity, paired differences, saturation, stability, repeated splits, DECISION.json
  04_FIGURES/                V2-1–V2-5 (SVG, PDF, PNG)
figures/                     manuscript figures 1–13 (SVG, PDF, PNG)
```

## Data

The raw EEG data are not included.

- **ds004504** (Miltiadous et al., 2023), OpenNeuro, v1.0.6: https://openneuro.org/datasets/ds004504
- **External cohort** (Shor et al., 2021; Benninger et al., 2022), Dryad, CC0:
  https://doi.org/10.5061/dryad.8gtht76pw. Only `alz_c1_new.mat` and `controls_c1_new.mat` are used.

## Environment

Python 3.11. Pinned versions are in `experiments/requirements.txt` (PyTorch with CUDA 12.6:
`--index-url https://download.pytorch.org/whl/cu126`). The external validation additionally needs
`h5py==3.16.0`; the ABBA reference check needs `fABBA==1.5.3`.

Paths are set with environment variables (defaults in `experiments/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `DS004504_ROOT` | `data/ds004504` | ds004504 root folder |
| `TEZ_CACHE` | `cache/` | intermediate files (about 6 GB) |

## Reproducing the results

```bash
# main pipeline (about 4–5 h; GPU used for the Transformer model)
cd experiments
bash run_all.sh            # 01 → 02 → {03, 04, 05, 06, 07, 08, 10} → 09 → 11

# supplementary analyses (after the main pipeline)
python ../supplementary_analyses/shape_scale_rd_bootstrap/a1_shape_scale_ablation.py
python ../supplementary_analyses/shape_scale_rd_bootstrap/a1_summarize.py
python ../supplementary_analyses/shape_scale_rd_bootstrap/a2_rd_bootstrap.py
python ../supplementary_analyses/abba_check_rd_bandwise/b1_abba_reference_check.py
python ../supplementary_analyses/abba_check_rd_bandwise/b2_rd_bandwise_and_delta_ci.py

# external validation (after the main pipeline)
cd ../external_validation/04_EXPERIMENTS/code
python 00_build_frozen_encoder.py
python 01_inspect_dryad.py  <folder with the Dryad .mat files>
python 02_prepare_dryad.py  <folder with the Dryad .mat files>
python 03_frozen_fidelity.py
python 04_classify_dryad.py
python 05_figures.py
python 06_results_table.py

# participant-disjoint recalibration (after 02_prepare_dryad.py); v2lib.py expects the first external
# analysis under the name EnvSAX_EXTERNAL_VALIDATION next to external_validation_v2/
cd ../../..
ln -s external_validation EnvSAX_EXTERNAL_VALIDATION     # Windows: mklink /D EnvSAX_EXTERNAL_VALIDATION external_validation
cd external_validation_v2/02_EXPERIMENTS/code
python 02_oof_calibration.py      # checks the sha256 lock before running
python 03_figures.py
```

`01_make_folds_and_lock.py` was run once to create the folds and the lock; rerunning it is not needed.

Evaluation is participant-level 5 × 5 repeated stratified cross-validation (`random_state=42`). Bootstrap
seeds and other settings are stated in each script.

## Notes on the external validation

The scale codebook learned on the main dataset was frozen (`03_PROTOCOL/FROZEN_ENCODER_INFO.json`, sha256) and
applied without modification. The protocol was fixed before the external EEG data were inspected; the amendment
(§11.1 of `EXTERNAL_PROTOCOL.md`) was written before any result was computed. The frozen codebook did **not**
preserve the group-difference profile in the external cohort (r_d = 0.804, below the channel-shuffle reference
0.810; band-mean-removed r_d = 0.118), whereas a label-free refit on that cohort did (r_d = 0.992). Under the
pre-specified rule the external check is *inconclusive* as a validation of the method.

A post hoc participant-disjoint check (`external_validation_v2/`), designed after these results were known but locked before its own results were computed, relearned the codebook without labels in five stratified folds (labels used only for stratification) and decoded each participant with a codebook that never saw its data. Out-of-fold r_d was 0.991 [0.967, 0.994], CCC 0.991 and band-mean-removed r_d 0.884 [0.794, 0.941], close to the full-cohort refit but below the protocol-defined criterion of 0.90 for band-mean-removed r_d; the pre-specified verdict is *partially supported*. This is a within-cohort calibration, not a frozen cross-site transfer. Participant-level window codes and decoded features are not included; the out-of-fold group profiles are.

## Citation

If you use this code, please cite the article (reference to be added on publication) and the datasets above.
See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The datasets keep their own licenses (ds004504: CC0; Dryad: CC0).

## Contact

İlhami Sel, Department of Computer Engineering, İnönü University, Malatya, Türkiye — ilhami.sel@inonu.edu.tr
