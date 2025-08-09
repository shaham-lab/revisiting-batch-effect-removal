import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
from dataset_reader.read_dataset_1 import get_dataset
from expirments.utils import make_combinations_from_config, sample_from_space, plot_adata
from main import ber_for_notebook
from expirments.load import load_to_adata_shaham_dataset, get_batch_from_adata, make_adata_from_batches

# from scDML.scDML.metrics import evaluate_dataset

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

print(f"here {os.curdir}")
plots_dir = r"../plots/ours/dataset-p1-3m/"
os.makedirs(plots_dir, exist_ok=True)

# Load your data into two numpy arrays
parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')

config = {
    "hidden_dim_encoder": [20],
    "hidden_dim_decoder": [20],
    "code_dim": [30],  #
    "drop_prob": [0.2,0.25],  # or tune.choice([<list values>])

    "lr": [ 0.01],  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2],  # or tune.choice([<list values>])
    "weight_decay": [0.2],  # or tune.choice([<list values>])
    "batch_size": [64, 128],  # or tune.choice([<list values>])
    "epochs": [40],
    "coef_1": [10],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset1-benchmark-new/")],
    "plots_dir": [os.path.join(parent_dir, "plots/mean_varience/dataset1-benchmark-new/")]
}
configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=1)

if __name__ == "__main__":
    adata = get_dataset()

    for config in configurations:
        os.makedirs(config["save_weights"], exist_ok=True)
        os.makedirs(config["plots_dir"], exist_ok=True)

        plot_adata(adata, plot_dir=config["plots_dir"], title='before-calibrationp')
        with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
            file.write(json.dumps(config))
        adata1 = adata[adata.obs['batch'] == 'Batch1', :].copy()
        adata2 = adata[adata.obs['batch'] == 'Batch2', :].copy()

        from baseline_methods.mean_varince import batch_effect_correction
        batch_a_to_b = batch_effect_correction(adata1.X,adata2.X)
        adata_target_calibrated_src = make_adata_from_batches(batch_a_to_b,adata2.X,adata1.obs['celltype'],adata2.obs['celltype'])
        plot_adata(adata_target_calibrated_src, plot_dir=config["plots_dir"],
                   title='after-calibration-target_calibrated_src')

        adata_target_calibrated_src.write(os.path.join(config["plots_dir"],'mean_varience_dataset_1.h5ad'))
