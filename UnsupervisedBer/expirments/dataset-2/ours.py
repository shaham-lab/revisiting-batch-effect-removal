import json
import os
from pathlib import Path

import numpy as np
import scib

from dataset_reader.read_dataset_2 import get_dataset
from expirments.load import assign_labels_to_numbers
from expirments.utils import sample_from_space, make_combinations_from_config, plot_adata
from main import ber_for_notebook

# from scDML.scDML.metrics import evaluate_dataset, silhouette_coeff_ASW

parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')

config = {

    "hidden_dim_encoder": [100,250],
    "hidden_dim_decoder": [100,250],
    "code_dim": [128],  #

    "lr": [0.001,0.01,0.1],  # np.random.uniform(1e-2, 1e-1, size=5).tolist(),  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2],  # or tune.choice([<list values>])
    "weight_decay": [0.2, 0.25, 0.3],  # or tune.choice([<list values>])
    "batch_size": [128, 64],  # or tune.choice([<list values>])
    "drop_prob": [0.2],  # or tune.choice([<list values>])
    "scale": [False],
    "hvg": [False],

    "epochs": [40],
    "coef_1": [1, 20, 50],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset2-benchmark-new-1/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset2-benchmark-new-1/")]
}

dim_reduce_weights_path = os.path.join(config["save_weights"][0], "dim_reduce")

configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=10)

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
            if config["scale"] == True:
                print("perform scale")
                adata = scib.preprocessing.scale_batch(adata, "batch")

            batch_name = list(set(adata.obs['batch']))
            adata1 = adata[adata.obs['batch'] == batch_name[0], :].copy()
            adata2 = adata[adata.obs['batch'] == batch_name[1], :].copy()

            os.makedirs(config["save_weights"], exist_ok=True)
            os.makedirs(config["plots_dir"], exist_ok=True)
            plot_adata(adata, plot_dir=config["plots_dir"], embed='X_pca', label='celltype', title='before-calibrationp')

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
            adata_target_calibrated_src.write(os.path.join(config["plots_dir"],'after_calibration_target_calibrated_src.h5ad'))
            adata_src_calibrated_target.write(os.path.join(config["plots_dir"] , 'after_calibration_src_calibrated_target.h5ad'))
            adata_code.write(os.path.join(config["plots_dir"] , 'code_1.h5ad'))
        except Exception:
            pass