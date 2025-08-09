import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import scanpy as sc
from dataset_reader.read_dataset_p1_baseline import get_dataset
from expirments.utils import make_combinations_from_config, sample_from_space, plot_adata
from expirments.load import load_to_adata_shaham_dataset, get_batch_from_adata, make_adata_from_batches, \
    assign_labels_to_numbers

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
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset-p1-baseline/")],
    "plots_dir": [os.path.join(parent_dir, "plots/pca_correct/dataset-p1-baseline/")]
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
        batch_name = list(set(adata.obs['batch']))
        adata1 = adata[adata.obs['batch'] == batch_name[0], :].copy()
        adata2 = adata[adata.obs['batch'] == batch_name[1], :].copy()

        from baseline_methods.pca_transform import batch_correction_pca
        corected_data,_ = batch_correction_pca(adata.X,np.array(assign_labels_to_numbers(adata.obs['batch'])[0]))
        adata_target_calibrated_src = sc.AnnData(X=corected_data)
        adata_target_calibrated_src.obs['batch'] = adata.obs['batch'].to_numpy()
        adata_target_calibrated_src.obs['celltype'] = adata.obs['celltype'].to_numpy()
        plot_adata(adata_target_calibrated_src, plot_dir=config["plots_dir"],
                   title='after-calibration-target_calibrated_src')

        adata_target_calibrated_src.write(os.path.join(config["plots_dir"],'pca_dataset_p1_baseline.h5ad'))
