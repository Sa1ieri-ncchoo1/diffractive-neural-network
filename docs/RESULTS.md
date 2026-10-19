# Evidence ledger

Evidence labels used here: **reported** means present in the supplied thesis/figures; **checkpoint-verified** means inference was rerun; **structurally verified** means calculated from actual tensors. None of these imply a full training reproduction or a hardware demonstration.

## Main experiment mapping

All checkpoint evaluations used the complete standard MNIST test split (10,000 images), square-root amplitude input, no evaluation augmentation, 532 nm wavelength, 8 um pixel pitch, 10 mm propagation distances and five trainable layers. Detector geometry was recovered from stored masks where available; the Pixel-MSE configuration without saved masks was recovered from supplied `test.py` and its filename.

| Included checkpoint | Original filename | Correct / 10,000 | Evaluation record |
|---|---|---:|---|
| `baseline_200.pt` | `d2nn_model_5layers.pth` | 8,947 | `validation/baseline_alternative_verified/metrics.json` |
| `contrast_200.pt` | `lr6e-3_15epoch_5layers_contrastloss_sumnorm` | 9,756 | `validation/contrast_200_verified/metrics.json` |
| `contrast_100.pt` | `lr6e-3_15epoch_5layers_contrastloss_sumnorm_ringdistance14_npixie100` | 9,730 | `validation/contrast_100_verified/metrics.json` |
| `contrast_100_ring20.pt` | `lr6e-3_15epoch_5layers_contrastloss_sumnorm_ringdistance20_npixie100` | 9,747 | `validation/contrast_100_ring20_verified/metrics.json` |
| `pixelmse_100.pt` | `lr6e-3_15epoch_5layers_pixelMSE_ringdistance20_npixie100` | 9,120 | `validation/pixelmse_100_verified/metrics.json` |

The first plausible baseline filename, `lr6e-3_15epoch_5layers`, actually scored **86.95%** under the matched saved geometry. That unsuccessful mapping attempt is retained in `validation/baseline_200_verified/`; its checkpoint hash distinguishes it from the included baseline. It was not silently rounded to or represented as 89.47%. The `baseline_alternative_verified` directory contains the correctly matched thesis baseline.

SHA-256 mappings are in `source_manifest.json`; each evaluation metric file independently records the loaded checkpoint hash. No physical buffers or phase tensors were altered to match a target score.

## Thesis claims and their status

The final thesis in the supplied archive is the primary reported source. PDF page numbers below are 1-based physical PDF pages; printed page numbers differ.

| Claim | Source location | Current status |
|---|---|---|
| 89.47% to 97.56% | Abstract, PDF p.3; chapter 3 table 3.2, PDF p.35 / printed p.27; supplied confusion matrices | Both endpoint checkpoint evaluations match; 8.09 percentage points |
| Parameters reduced 75% | Section 3.8.1, table 3.3, PDF p.38 / printed p.30 | Actual trainable tensors: 5×200²=200,000 and 5×100²=50,000; reduction verified |
| Accuracy decreased by 0.26 percentage points | Same section/table | 97.56% versus 97.30%, confirmed by checkpoint evaluation |
| Training approximately 4× faster | Same section/table, relative training time 1.0 versus 0.25 | Reported only; no original epoch timing log or controlled new training benchmark |
| Ring-radius variant 97.47% | Archive folder `contrastloss 100@100 ringdistance=20 97.47%` | Checkpoint evaluation matches |
| Pixel-MSE 91.20% | Table 3.2 and corresponding archive figure folder | Checkpoint evaluation matches |
| FDTD best approximately 11% | Chapter 4, PDF pp.45-50 | Reported only; explicitly 1/9 selected/diagnostic samples; no FDTD rerun |

The source archive is kept locally outside this package to avoid publishing personal academic information. Historical image paths and hashes are recorded in the manifest.

## Discrepancies and interpretation limits

1. The current original `d2nn.py` uses 56 pixels, 2 layers and 0.3 um pitch; it is not the old five-layer 200-pixel training configuration. `train.py` currently uses 56 pixels, 1 layer and Pixel-MSE. Historical training hyperparameters cannot all be inferred from checkpoint filenames.
2. The 200-pixel checkpoint masks have a 28-pixel ring radius and 8-pixel detector radius; the 100-pixel compression checkpoint has 14 and 4 pixels. The thesis paragraph says other parameters including a ring radius of 14 were unchanged. Saved masks are the stronger evidence for replaying these checkpoints.
3. At fixed 8 um pitch, halving detector radius from 8 to 4 pixels also halves its physical radius. The thesis phrase implying approximate preservation of physical detector size is inconsistent with those numbers. The compression comparison changes aperture and detector geometry as well as phase-parameter count; it is not an isolated parameter-count ablation.
4. Two raster pixels overlap between detector masks in the 100-pixel/ring-14 model. This historical behavior is preserved; changing it would change the experiment. The sum-normalized scores are sums of potentially overlapping regions, not a partition of all optical power.
5. The supplied implementation resizes MNIST to the full N×N field, then pads it during each propagation. Thesis text describing input placed only in the central N/2 region is not the same operation.
6. A chapter summary contains “87.47%” while the abstract, main table, saved historical figure and checkpoint evaluation support 89.47%. This is treated as a thesis inconsistency, not a second baseline.
7. Full-resolution training curves were not found in the supplied material. Do not infer or synthesize an epoch-by-epoch curve from final accuracies. The two-epoch, 64-sample functional run is recorded in `validation/training_subset_verified/`; it is not a historical training curve or convergence result. Its plot is omitted from this public repository.
8. No raw logs or uncertainty estimates establish the roughly 4× training-speed claim. Checkpoint replay and small training checks do not replace multi-seed training and controlled timing.
