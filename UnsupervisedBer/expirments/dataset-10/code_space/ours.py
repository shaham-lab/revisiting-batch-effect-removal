import json
import os
from pathlib import Path

import numpy as np
import scanpy as sc

from dataset_reader.read_dataset_5 import get_dataset
from expirments.utils import sample_from_space, make_combinations_from_config, plot_adata, plot_umap_celltype, \
    plot_umap_batch
from main import ber_for_notebook


from scDML.scDML.metrics import evaluate_dataset, silhouette_coeff_ASW

parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')

config = {
    "hidden_dim_encoder": [20],
    "hidden_dim_decoder": [100],
    "code_dim": [64],        # Dimension of the encoded code representation

    "lr": [0.01],  # np.random.uniform(1e-2, 1e-1, size=5).tolist(),  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "drop_prob": [0.2],  # or tune.choice([<list values>])
    "weight_decay": [0.2, 0.35, 0.3],  # or tune.choice([<list values>])
    "batch_size": [128, 64, 36],  # or tune.choice([<list values>])
    "epochs": [100],
    "coef_1": [1, 2, 5],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset5-benchmark/code_space")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset5-benchmark/code_space")]
}

dim_reduce_weights_path = os.path.join(config["save_weights"][0], "dim_reduce")

configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=10)

if __name__ == "__main__":
    adata_dim_reduce = get_dataset()
    print("----before----")
    # evaluate_dataset(adata_dim_reduce)
    adata1 = adata_dim_reduce[adata_dim_reduce.obs['batch'] == 1, :].copy()
    adata2 = adata_dim_reduce[adata_dim_reduce.obs['batch'] == 2, :].copy()

    # source, target, model_shrinking = pre_processing(adata1.X, adata2.X, num_epochs=100,
    #                                                  save_weights_path=dim_reduce_weights_path)
    # adata1.obsm["dim_reduce"], adata2.obsm["dim_reduce"] = source, target

    for config in configurations:
        os.makedirs(config["save_weights"], exist_ok=True)
        os.makedirs(config["plots_dir"], exist_ok=True)
        plot_adata(adata_dim_reduce, plot_dir=config["plots_dir"], embed='X_pca', label='celltype', title='before-calibrationp')

        with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
            file.write(json.dumps(config))

        silhouette_coeff_ASW(adata_dim_reduce).to_csv(
            os.path.join(config["plots_dir"], "ASW_orignal_adata.csv"))
        evaluate_dataset(adata_dim_reduce).to_csv(
            os.path.join(config["plots_dir"], "orignal_adata.csv"))  # Set the batch key for each cell

        code_src, code_target = ber_for_notebook(config, adata1=adata1,
                                                 adata2=adata2,
                                                 return_in="code_space",
                                                 embed='' )

        adata_code_space = sc.AnnData(X=np.concatenate((code_src.detach().numpy(), code_target.detach().numpy()), axis=0))
        adata_code_space.obs['celltype'] = np.array(adata_dim_reduce.obs['celltype'])
        adata_code_space.obs['batch'] = np.array(adata_dim_reduce.obs['batch'])

        plot_umap_celltype(adata_code_space, path=config["plots_dir"],
                           title='after-calibration-target_calibrated_src')
        plot_umap_batch(adata_code_space, path=config["plots_dir"],
                           title='after-calibration-src_calibrated_target')


