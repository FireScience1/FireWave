# FireWave

FireWave is a conditional wavelet-domain forecasting model for the FIT
(Fire in Tunnel) benchmark. It forecasts future temperature and visibility
fields from historical fields and tunnel-fire conditions.

This repository contains the FIT data interface, model and evaluation code,
and the parameter configurations used for the reported experiments.

## Repository layout

```text
FireWave/
├── configs/
│   ├── fit_direct_model.json
│   ├── fit_temperature.json
│   └── fit_visibility.json
├── data/
│   └── README.md
└── firewave/
    ├── core.py
    ├── direct_predictor.py
    ├── metrics.py
    ├── pixel_prior.py
    ├── postprocess.py
    ├── scenario_prior.py
    ├── strict_temperature_eval.py
    ├── strict_visibility.py
    ├── strict_visibility_eval.py
    ├── temperature_prior.py
    └── visibility_encoding.py
```

## Data

The code expects the FIT data directory to contain scenario folders such as
`A1`--`A4` and `B1`--`B4`. Each scenario should provide temperature,
visibility, and heat-release-rate data. See [data/README.md](data/README.md)
for the expected layout.

The raw FIT archive is approximately 4.2 GB and is tracked with Git LFS.
Install Git LFS and run `git lfs pull` after cloning to materialize the data.
Alternatively, set `data_root` in a configuration file or pass a local/HDFS
data path through the command line.

## Model and evaluation entry points

The main reusable components are:

- `firewave.core`: FIT storage, segment construction, normalization, and
  metric utilities.
- `firewave.direct_predictor`: direct conditional spatio-temporal predictor.
- `firewave.temperature_prior`: temperature-condition prior and evaluation.
- `firewave.strict_visibility`: visibility model training.
- `firewave.strict_temperature_eval` and `firewave.strict_visibility_eval`:
  leakage-audited evaluation entry points.
- `firewave.metrics`: MSE, SSIM, and PSNR aggregation.

All evaluation scripts use the validation split for model selection. The
held-out test split must remain untouched until the final evaluation.

## Parameter configurations

The configuration files record the FIT setup and model parameters:

- [fit_direct_model.json](configs/fit_direct_model.json): direct model
  architecture and optimization parameters.
- [fit_temperature.json](configs/fit_temperature.json): temperature-task
  condition and interpolation parameters.
- [fit_visibility.json](configs/fit_visibility.json): visibility-task
  training and evaluation parameters.

The default protocol uses 16 historical frames, predicts 32 future frames,
and uses split seed `1`. Metrics are computed after per-field global min--max
normalization with `data_range=1`.

## Running the code

From the repository root, after installing PyTorch and the dependencies used
by the selected entry point:

```bash
python -m firewave.temperature_prior \
  --data-root /path/to/FDS_fire \
  --output temperature_results \
  --save-arrays temperature_arrays

python -m firewave.strict_visibility \
  --data-root /path/to/FDS_fire \
  --output-dir visibility_results
```

For strict evaluation, provide the selected validation configuration or
checkpoint explicitly and keep the test split inaccessible during selection.
The command-line options are exposed by each module's `--help` output.

## Reproducibility notes

- Use the configuration files as the source of truth for model parameters.
- Keep the split seed fixed when reproducing the reported protocol.
- Do not commit raw data, checkpoints, generated arrays, or machine-specific
  absolute paths.
- The repository is intended to be read and reused by reviewers without
  requiring access to the original development environment.

## License and citation

The repository is released for peer-review and reproducibility purposes.
Please cite the accompanying FireWave manuscript when using the code.
