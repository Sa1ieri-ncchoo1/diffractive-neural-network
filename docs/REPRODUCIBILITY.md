# Environment and reproduction

## What was run

Validation date: 2026-10-12 (Asia/Hong_Kong).

| Component | Observed local environment |
|---|---|
| Python | 3.10.20, Anaconda |
| PyTorch | 2.11.0.dev20260203+cu128 |
| torchvision | 0.25.0.dev20260205+cu128 |
| NumPy | 2.0.1 |
| Matplotlib | 3.9.1 |
| CUDA runtime reported by torch | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| OS | Windows, build 26200 |

The exact observed package snapshot is saved in `validation/environment.json`. This is an existing nightly environment, not a clean-environment installation test. Those historical nightly wheels may not remain available. `requirements.txt` lists four direct dependencies without version pins; it is not a proven cross-platform lockfile. A new environment must install a compatible torch/torchvision pair and pass the smoke checks before benchmarking. CPU evaluation is supported by the CLI but only synthetic CPU forward/backward checks were executed here; full-test-set measurements used CUDA.

## Installation and working directory

Run all commands from the repository root. `python` below means the interpreter in your chosen environment. The supported CLI has no dependency on the original local source path.

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m d2nn.cli smoke --config configs/current_56.json --device cpu --output runs/cpu_smoke
python -m unittest discover -s tests -v
```

No dataset or network access is needed for these smoke/unit tests. Tests require the included 100-pixel checkpoint. Avoid arbitrary old pickle checkpoints; the loader accepts state dictionaries with `weights_only=True`.

## Evaluate historical checkpoints

```bash
python -m d2nn.cli evaluate --config configs/baseline_200.json --data-root data --download --output runs/baseline
python -m d2nn.cli evaluate --config configs/contrast_200.json --data-root data --output runs/contrast200
python -m d2nn.cli evaluate --config configs/contrast_100.json --data-root data --output runs/contrast100
python -m d2nn.cli evaluate --config configs/contrast_100_ring20.json --data-root data --output runs/ring20
python -m d2nn.cli evaluate --config configs/pixelmse_100.json --data-root data --output runs/pixelmse
```

MNIST is obtained through torchvision and excluded from the archive. The expected split sizes are 60,000 train and 10,000 test; `validation/dataset_manifest.json` records local data hashes. Without `--download`, missing data raises an error instead of accessing the network. Use a chosen external dataset location via `--data-root` to avoid duplicate copies.

Each run records checkpoint SHA-256, environment, sample count, correct count, loss, configuration, a CSV confusion matrix and a PNG. No `--limit` is used for published checkpoint results. A limit evaluates the first N examples and is useful only for functional checks. Evaluation uses no augmentation and no added noise. New models use seed 42. Exact results on different devices/library builds are not guaranteed; preserve logs and investigate differences instead of editing the claimed accuracy.

## Train and validate new checkpoints

For a quick end-to-end check:

```bash
python -m d2nn.cli train --config configs/current_56.json --data-root data --download --epochs 2 --limit 64 --batch-size 16 --output runs/training_smoke
python -m d2nn.cli evaluate --config runs/training_smoke/config.json --checkpoint runs/training_smoke/model.pt --data-root data --limit 64 --batch-size 16 --output runs/reload_smoke
```

For a full new run, omit `--limit`, use a thesis-size config and record the entire history:

```bash
python -m d2nn.cli train --config configs/contrast_100.json --data-root data --epochs 15 --batch-size 256 --output runs/full_new_training
```

Reduce the batch size if GPU memory is insufficient, but record that this changes optimization. The historical full batch size is not fully proven by the source snapshot; 256 is present in surviving training scripts. Evaluation at batch size 32 does not reconstruct the original training batch size. Configurations retain the supplied phase initialization, propagation definitions and losses; reconstructed augmentation is clearly marked in each JSON. Changing loss, augmentation, encoding or geometry creates a different experiment.

The CLI saves the final epoch, not a checkpoint selected by test accuracy. It evaluates the test set each epoch to retain the original development pattern; for future model selection, create a separate validation split and reserve the test set for final reporting. It does not save optimizer state or support exact resume. Python, NumPy and torch seeds are set; this is not a guarantee of deterministic behavior on every backend.

## Checks completed and limits

- Original core definitions compared by Python AST: all eight extracted definitions match.
- Five functional tests: zero-distance propagation; gradients through every phase layer for all three losses; expected contrast-loss margin; detector mismatch rejection; physical-buffer mismatch rejection.
- CPU synthetic forward/backward and optimizer-step smoke check passed.
- CUDA training on 64 training images for two epochs, evaluation on 64 test images, checkpoint save/reload and plotting passed. Reloaded loss and correct count matched. This is deliberately too small to support a benchmark claim.
- Five historical checkpoint evaluations on all 10,000 test images matched the reported metrics.
- Fresh environment installation, full training from scratch, multi-seed variation, training-speed benchmarking and all Lumerical execution remain unverified.

## Known environment and legacy issues

The original local environment exposed a duplicate OpenMP-runtime error when plotting after GPU numerical work. The supported entry point saves metrics first and launches plotting in a separate process; no unsafe duplicate-runtime override is enabled. If this still fails on another environment, numeric metrics remain saved and `python -m d2nn.plotting RUN_DIRECTORY` can regenerate figures after the environment is corrected.

Historical standalone scripts are excluded from this public edition. They contained import-time side effects and incomplete FDTD dependencies. FDTD requires licensed Lumerical, original helper modules and complete simulation configuration; this repository supplies a limitations report, not an executable FDTD pipeline.
