import json
import os
import random
import time
from pathlib import Path

import numpy as np
import scib
import torch
from toolz import excepts

from dataset_reader.read_dataset_5 import get_dataset
from metrics import eval_mmd

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from expirments.load import assign_labels_to_numbers
from expirments.utils import make_combinations_from_config, sample_from_space, plot_adata
from main import ber_for_notebook

data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\batch_effect\dataset5"

parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')

config = {
    "hidden_dim_encoder": [512],
    "hidden_dim_decoder": [512],
    "code_dim": [256],  #

    "lr": [0.01, 0.001],  # np.random.uniform(1e-2, 1e-1, size=5).tolist(),  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2],  # or tune.choice([<list values>])
    "weight_decay": [0.2, 0.35, 0.3],  # or tune.choice([<list values>])
    "batch_size": [128, 64],  # or tune.choice([<list values>])
    "drop_prob": [0.2],  # or tune.choice([<list values>])
    "scale": [False, True],
    "hvg": [False, True],

    "epochs": [40],
    "coef_1": [1, 2, 5],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset5-benchmark/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset5-benchmark/")]
}
dim_reduce_weights_path = os.path.join(config["save_weights"][0], "dim_reduce")
os.makedirs(dim_reduce_weights_path, exist_ok=True)
configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=7)
# configurations = [configurations[1]]
if __name__ == "__main__":
    adata_orignal = get_dataset()
    encoding_labels,number_to_label = assign_labels_to_numbers(adata_orignal.obs['celltype'])
    adata_orignal.obs['celltype'] = np.array(encoding_labels)

    # Set the batch key for each cell

    for config in configurations:
        try:
            adata = adata_orignal.copy()
            if config["hvg"]==True:
                adata = scib.preprocessing.hvg_batch(
                    adata,
                    batch_key="batch",
                    target_genes=2000,
                    adataOut=True
                )
            if True==True:
                print("perform scale")
                adata = scib.preprocessing.scale_batch(adata, "batch")


            adata1 = adata[adata.obs['batch'] == "0", :].copy()
            adata2 = adata[adata.obs['batch'] == "1", :].copy()

            os.makedirs(config["save_weights"], exist_ok=True)
            os.makedirs(config["plots_dir"], exist_ok=True)
            plot_adata(adata, plot_dir=config["plots_dir"], embed='X_pca', label='celltype', title='before-calibrationp')

            with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
                file.write(json.dumps(config))

            adata_target_calibrated_src, adata_src_calibrated_target = ber_for_notebook(config, adata1=adata1,
                                                                                        adata2=adata2,
                                                                                        return_in="original_space",
                                                                                        embed='')

            plot_adata(adata_target_calibrated_src, plot_dir=config["plots_dir"],
                       title='after-calibration-target_calibrated_src')
            plot_adata(adata_src_calibrated_target, plot_dir=config["plots_dir"],
                       title='after-calibration-src_calibrated_target')

            adata_target_calibrated_src.write(os.path.join(config["plots_dir"],'after_calib_src_target.h5ad'))
            adata_src_calibrated_target.write(os.path.join(config["plots_dir"] , 'after_calib_target_src.h5ad'))
        except Exception as e:
            pass
