## Revisiting Batch Effect Removal (BER)

This repository contains two complementary methods for single‑cell batch effect removal:

- SupervisedBer: supervised domain adaptation using labeled cell types across batches.
- UnsupervisedBer: an autoencoder with a batch‑conditioned decoder and an independence discriminator to align batches without labels.

Both methods operate on `AnnData` objects (Scanpy) and provide example experiment scripts over multiple datasets.

### Repository layout

- `SupervisedBer/` — supervised method, training and experiment scripts under `expirements/`.
- `UnsupervisedBer/` — unsupervised method, core model in `unsupervised/`, experiments in `expirments/`.
- `SupervisedBer/dataset_reader/` and `UnsupervisedBer/dataset_reader/` — dataset loaders (paths may need updating for your machine).

### Method overview

- SupervisedBer
  - Learns a shared embedding via a classifier `Net` trained on concatenated batches.
  - Loss combines source and target supervised terms with optional unsupervised alignment (e.g., MMD) and a cross‑domain cell‑type alignment (CDCA) term.
  - Produces a batch‑aligned latent representation (`X_emb`) that preserves cell‑type structure.

- UnsupervisedBer
  - Encoder–Decoder autoencoder with a batch‑conditioned decoder (`DecoderBer`).
  - A discriminator encourages batch‑invariant latent codes; calibration is done by decoding with the opposite batch indicator.
  - Uses MMD/UMAP/PCA for monitoring and returns calibrated data in original space and/or the code space.

### Requirements

Create a fresh environment (Python 3.8–3.10 recommended). Then install:

```
pip install -r requirements.txt
```

Files provided:

- `requirements.txt` — includes both method requirements via `-r` includes
- `requirements_supervised.txt` — dependencies for `SupervisedBer`
- `requirements_unsupervised.txt` — dependencies for `UnsupervisedBer`
- `requirements_extras.txt` — optional baselines (e.g., LIGER, Harmony)

If you use GPU, install a CUDA‑compatible PyTorch per the official guidance before running `pip install -r ...`.

### Data preparation

The dataset loaders currently point to example local paths on Windows. Before running, update the paths inside:

- `SupervisedBer/dataset_reader/read_dataset_*.py`
- `UnsupervisedBer/dataset_reader/read_dataset_*.py`

Each loader expects gene‑by‑cell expression and metadata with at least `batch` and `celltype` (names may vary per script). You can also adapt the readers to load `.h5ad` files you already have.

Tip: The `UnsupervisedBer/dataset_reader/` folder contains example `.h5ad` files you can use to validate the pipeline after adjusting readers.

### How to run

All examples below assume you run from the method’s directory.

- UnsupervisedBer
  1) Change directory: `cd UnsupervisedBer`
  2) Run an experiment script, for example dataset 2:
     - `python expirments/dataset-2/ours.py`
     - The script defines a search space in its `config` dict and writes results to `expirments/plots/...` and weights to `expirments/weights/...`.
  3) Core function reference: see `main.py: ber_for_notebook`, which returns calibrated `AnnData` objects (code space and/or original space).

- SupervisedBer
  1) Change directory: `cd SupervisedBer`
  2) Run a supervised alignment experiment, for example dataset 2:
     - `python expirements/dataset2/ours.py`
     - Results (plots, `.h5ad` outputs) are written under `expirements/plots/...`; model weights under `expirements/weights/...`.

Notes

- Some experiment folders have names like `dataset-2` (with a hyphen). Invoke these scripts by file path (as shown) rather than `python -m ...`.
- TensorBoard logs (for supervised training) are written under `runs/` in the corresponding experiment directory.

### Outputs

- Calibrated `AnnData` in original space: `after_calibration_*.h5ad`
- Latent/code `AnnData`: `code_*.h5ad` or `cdca_latent_*.h5ad`
- Plots: UMAP/PCA colored by batch and celltype in `expirments/plots/...`

### Reproducibility

Many scripts set random seeds (`numpy`, `torch`, `random`). For full determinism across hardware/backends you may additionally set PyTorch deterministic flags as needed.

### Troubleshooting

- Import errors for baselines (e.g., `pyliger`, `harmonypy`): install `requirements_extras.txt`.
- GPU/CPU selection: the unsupervised code auto‑detects CUDA; you can force CPU by setting the `CUDA_VISIBLE_DEVICES` environment variable to empty.
- Scanpy/H5AD I/O: if you encounter HDF5 errors, ensure a compatible `h5py` is installed (it is a transitive dependency of `anndata`).


