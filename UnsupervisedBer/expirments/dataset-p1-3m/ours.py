import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import scanpy as sc

from baseline_methods.mean_varince import batch_effect_correction
from dataset_reader.read_dataset_p1_3m import get_dataset
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
    "hidden_dim_encoder": [50, 100],
    "hidden_dim_decoder": [50, 100],
    "code_dim": [30, 40],  #
    "drop_prob": [0.2, 0.25],  # or tune.choice([<list values>])

    "lr": [0.001, 0.005],  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2, 0.15, 0.1],  # or tune.choice([<list values>])
    "weight_decay": [0.2, 0.15],  # or tune.choice([<list values>])
    "batch_size": [64, 128],  # or tune.choice([<list values>])
    "epochs": [40],
    "coef_1": [10, 1, 5],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset-p1-3m_new/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset-p1-3m_new/")]
}
configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=10)

if __name__ == "__main__":
    adata = get_dataset()

    for config in configurations:
        os.makedirs(config["save_weights"], exist_ok=True)
        os.makedirs(config["plots_dir"], exist_ok=True)
        # evaluate_dataset(adata).to_csv(
        #     os.path.join(config["plots_dir"], "orignal_adata.csv"))  # Set the batch key for each cell
        #
        plot_adata(adata, plot_dir=config["plots_dir"], title='before-calibrationp')
        with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
            file.write(json.dumps(config))
        adata1, adata2 = get_batch_from_adata(adata)
        batch_a_to_b = batch_effect_correction(adata1.X, adata2.X)
        adata_orignal = make_adata_from_batches(batch_a_to_b, adata2.X, adata1.obs['celltype'],
                                                adata2.obs['celltype'])
        adata1, adata2 = get_batch_from_adata(adata_orignal)

        # sc.pp.scale(adata1)
        # sc.pp.scale(adata2)

        adata_code, adata_target_calibrated_src, adata_src_calibrated_target = ber_for_notebook(config, adata1=adata1,
                                                                                                adata2=adata2,
                                                                                                return_in="original_space_and_code")
        plot_adata(adata_target_calibrated_src, plot_dir=config["plots_dir"],
                   title='after-calibration-target_calibrated_src')
        plot_adata(adata_src_calibrated_target, plot_dir=config["plots_dir"],
                   title='after-calibration-src_calibrated_target')
        plot_adata(adata_code, plot_dir=config["plots_dir"],
                   title='cdoe_space')

        adata_target_calibrated_src.write(os.path.join(config["plots_dir"], 'after_calib_src_target.h5ad'))
        adata_src_calibrated_target.write(os.path.join(config["plots_dir"], 'after_calib_target_src.h5ad'))
        adata_code.write(os.path.join(config["plots_dir"], 'code_1.h5ad'))
        # evaluate_dataset(adata_src_calibrated_target).to_csv(
        #     os.path.join(config["plots_dir"], "adata_src_calibrated_target.csv"))  # Set the batch key for each cell
        # evaluate_dataset(adata_target_calibrated_src).to_csv(
        #     os.path.join(config["plots_dir"], "adata_target_calibrated_src.csv"))  # Set the batch key for each cell
