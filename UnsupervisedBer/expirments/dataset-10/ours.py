import json
import os
import random
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import pandas as pd
import torch

from dataset_reader.read_dataset_10 import get_dataset
from metrics import eval_mmd

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from expirments.load import assign_labels_to_numbers, get_batch_from_adata
from expirments.plot_benchmark import plotTSNE, plotUMAP, save_output_csv
from expirments.utils import make_combinations_from_config, sample_from_space, plot_adata
from main import ber_for_notebook


data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\batch_effect\dataset10"
parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')

config = {
    "hidden_dim_encoder": [100,200],
    "hidden_dim_decoder": [100,200],
    "code_dim": [256],  #

    "lr": [0.001,0.01],  # np.random.uniform(1e-2, 1e-1, size=5).tolist(),  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2,0.25],  # or tune.choice([<list values>])
    "weight_decay": [0.3,0.25],  # or tune.choice([<list values>])
    "batch_size": [128,256],  # or tune.choice([<list values>])
    "drop_prob": [0.2,0.25],  # or tune.choice([<list values>])

    "epochs": [40],
    "coef_1": [2,1,0.5],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset10-benchmark-new-1/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset10-benchmark-new-1/")]
}
dim_reduce_weights_path = os.path.join(config["save_weights"][0], "dim_reduce")
os.makedirs(dim_reduce_weights_path, exist_ok=True)
configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=10)
# configurations = [configurations[1]]
if __name__ == "__main__":
    adata = get_dataset()
    encoding_labels,number_to_label = assign_labels_to_numbers(adata.obs['celltype'])
    adata.obs['celltype'] = np.array(encoding_labels)

    # Set the batch key for each cell
    adata1 = adata[adata.obs['batch'] == '1', :].copy()
    adata2 = adata[adata.obs['batch'] == '2', :].copy()
    print(set(adata1.obs['celltype']))
    print(set(adata2.obs['celltype']))



    for config in configurations:
        os.makedirs(config["save_weights"], exist_ok=True)
        os.makedirs(config["plots_dir"], exist_ok=True)
        plot_adata(adata, plot_dir=config["plots_dir"], embed='X_pca', label='celltype', title='before-calibrationp')
        #
        # sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)

        with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
            file.write(json.dumps(config))


        adata_code,adata_target_calibrated_src, adata_src_calibrated_target = ber_for_notebook(config, adata1=adata1,
                                                                                    adata2=adata2,return_in="original_space_and_code")
        plot_adata(adata_target_calibrated_src, plot_dir=config["plots_dir"],
                   title='after-calibration-target_calibrated_src')
        plot_adata(adata_src_calibrated_target, plot_dir=config["plots_dir"],
                   title='after-calibration-src_calibrated_target')
        plot_adata(adata_code, plot_dir=config["plots_dir"],
                   title='cdoe_space')

        adata_target_calibrated_src.write(os.path.join(config["plots_dir"],'after_calib_src_target.h5ad'))
        adata_src_calibrated_target.write(os.path.join(config["plots_dir"] , 'after_calib_target_src.h5ad'))
        adata_code.write(os.path.join(config["plots_dir"] , 'code_1.h5ad'))
