# FDTD validation: unsuccessful, diagnostic work only

This project does **not** demonstrate successful physical validation of the MNIST classifier. Lumerical FDTD was not run during this cleanup. The reviewed historical code included preparation and comparison scripts, but the necessary `tools.structure` / `tools.utils` helpers and complete per-run simulation projects are absent from the reviewed source directory.

## Intended method

1. Train an ASM phase-only model in PyTorch.
2. Export phase using `phi = pi*tanh(raw_phase)`, wrap into `[0,2*pi)`, and export `(H,W,L)` arrays.
3. Map phase to a dielectric height map and construct a Lumerical simulation.
4. Encode MNIST as a physical input mask or imported complex source.
5. Read detector-plane fields, form intensity, align coordinates and integrate class detector regions.
6. Compare identical inputs and geometries with the PyTorch fields.

The supplied export default refers to a checkpoint filename containing `wavelength266`, while several simulation defaults are 532 nm. A filename is not sufficient evidence of actual wavelength. Physical buffer checks, manifest metadata and export configuration must agree before any rerun.

## What the thesis actually reports

The final thesis, chapter 4 (PDF pages 45-50; printed pages 37-42), reports:

| Experiment | Configuration/changes described | FDTD result | Evidence scope |
|---|---|---|---|
| Physical input mask (v3 in thesis) | N=56, pixel pitch 0.3 um, wavelength 532 nm, distance 8.75 um, 2 layers, refractive index 1.7227 | 0/8 | Eight diagnostic samples |
| Imported source (v4 in thesis) | Input injection changed; layer count also reduced to 1 | 1/9 (11.1%) | Nine diagnostic samples |
| High-index exploratory variant | Refractive index 8, smaller height range, input changed to 10x10 with surrounding blocking region | 1/9 | Nine diagnostic samples |

The 11% figure is **not** accuracy on the complete 10,000-image MNIST test split. The repository includes easy-example selection code, so selection bias must also be considered. The main 97.56% ASM thesis result uses a different geometry and cannot be treated as a matched comparator to these nine samples.

The thesis gives varying ASM comparators (approximately 95%, 56%, and 40%) across prose/table/variants. Do not merge them into one headline gap. The script called `main_mnist_v3.py` uses imported sources, while the thesis labels that approach v4: filenames alone do not uniquely identify the experiment.

## Unresolved interpretation

The author proposes mismatch between scalar ASM with thin phase modulation and the subwavelength dielectric geometry as an explanation. This remains a hypothesis supported by diagnostic observations, not a uniquely isolated cause established by the supplied material. Multiple factors changed between comparisons: input representation, layer count, refractive index and input resolution. Small sample sizes and missing raw field/mesh records further limit inference.

Avoid claims that all engineering causes have been ruled out, or that ASM and FDTD are universally incompatible. The reviewed evidence cannot establish those conclusions. Likewise, a fixed wavelength-to-grid ratio alone does not establish FDTD convergence or temporal stability.

## Required before a meaningful rerun

- Recover the exact scripts/helpers, Lumerical version, material definitions, boundary conditions, source polarization and mesh/time convergence records.
- Use the same checkpoint, wavelength, pitch, distances, layer count and input field on both sides; retain a machine-readable manifest and hashes.
- Verify transpose/flip/rotation conventions on a calibration field before viewing classification labels. Do not choose coordinate transforms to maximize accuracy on the evaluated samples.
- Test propagation without masks, one uniform phase layer and one calibrated nonuniform layer before full classification.
- Check source-field normalization, phase wrapping, height mapping, monitor position and aperture.
- Change one factor at a time; use a fixed prespecified sample list with reported selection criteria and sample counts.
- Report field/intensity errors and convergence alongside classification; retain unsuccessful runs.

The incomplete historical FDTD scripts are excluded from this public edition. Only the portable ASM/MNIST workflow is supported. No simulation license, helper code or successful physical result is supplied by this package.
