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
from metric import silhouette_coeff_ASW
from train_sda import cdca_alignment

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from expirements.load import assign_labels_to_numbers
from expirements.plot_benchmark import plotTSNE, plotUMAP, save_output_csv
from expirements.utils import make_combinations_from_config, sample_from_space, plot_adata
from pre_procesing.train_reduce_dim import pre_processing


data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\batch_effect\dataset10"

parent_dir = Path(r'C:/Users/avrah/PycharmProjects/cdca/expirements')

config = {
    "loss_type": ["s&t&u&c"],
    "experiment_name": ["s&t&u&c"],
    "hidden_dim": [30],
    "drop_prob": [0.2,0.25],
    "hidden_layers": [10,12,8],
    "lr": [0.01,0.001,0.0001],  # np.random.uniform(1e-2, 1e-1, size=5).tolist(),  # Nuber of covariates in the data
    "test_size": [0.3],
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2,0.3],  # or tune.choice([<list values>])
    "weight_decay": [0.2,0.25,0.3],  # or tune.choice([<list values>])
    "batch_size": [128,256,512],  # or tune.choice([<list values>])
    "epochs": [120],
    "coef":[1,10,20],
    "coef_uda": [1, 2, 5],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset10-benchmark/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset10-benchmark-fixscale/")]
}
dim_reduce_weights_path = os.path.join(config["save_weights"][0], "dim_reduce")
os.makedirs(dim_reduce_weights_path, exist_ok=True)
configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=14)
# configurations = [configurations[1]]


if __name__ == "__main__":
    random.seed(3)
    np.random.seed(3)

    adata = get_dataset()
    # Set the batch key for each cell
    encoding_labels,number_to_label = assign_labels_to_numbers(adata.obs['celltype'])
    adata.obs['encoding-celltype'] = np.array(encoding_labels)

    adata1 = adata[adata.obs['batch'] == '1', :].copy()
    adata2 = adata[adata.obs['batch'] == '2', :].copy()

    for config in configurations:

        os.makedirs(config["save_weights"], exist_ok=True)
        os.makedirs(config["plots_dir"], exist_ok=True)
        plot_adata(adata,embed="dim_reduce", plot_dir=config["plots_dir"],title='before')
        #silhouette_coeff_ASW(adata, embed='').to_csv(os.path.join(config["plots_dir"],
        #                                                          "ASW_adata_before.csv"))


        with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
            file.write(json.dumps(config))

        adata_calibration = cdca_alignment(config, adata1=adata1, adata2=adata2,number_to_label=number_to_label)

        adata_calibration.write(os.path.join(config["plots_dir"],'cdca_latent_1.h5ad'))
