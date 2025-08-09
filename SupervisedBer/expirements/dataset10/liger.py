import os
import random
from pathlib import Path

import pandas as pd

import pyliger
import scanpy as sc
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from expirements.load import assign_labels_to_numbers
from expirements.utils import make_combinations_from_config, sample_from_space
from metric import silhouette_coeff_ASW
from plot_data import plot_umap_batch, plot_umap_celltype
from pre_procesing.train_reduce_dim import pre_processing

data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\batch_effect\dataset10"

# Load your data into two numpy arrays
parent_dir = Path(r'C:/Users/avrah/PycharmProjects/cdca/expirements')

config = {
    "loss_type": ["s&t&u&c"],
    "experiment_name": ["s&t&u&c-50"],
    "input_dim": [25],
    "hidden_dim": [30],
    "drop_prob": [0.2],
    "hidden_layers": [5],
    "lr": [0.01],  # np.random.uniform(1e-2, 1e-1, size=5).tolist(),  # Nuber of covariates in the data
    "test_size": [0.85],
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2],  # or tune.choice([<list values>])
    "weight_decay": [0.2],  # or tune.choice([<list values>])
    "batch_size": [500],  # or tune.choice([<list values>])
    "epochs": [150],
    "coef": [1, 10, 100],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset10-benchmark/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset10-benchmark/")]
}
dim_reduce_weights_path = os.path.join(config["save_weights"][0], "dim_reduce")
os.makedirs(dim_reduce_weights_path, exist_ok=True)
configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=3)
# configurations = [configurations[1]]
def setX(adata,x):
    temp_adata = sc.AnnData(X=x)
    temp_adata.obs['batch'] = np.array(adata.obs['batch'])
    temp_adata.obs['celltype'] = np.array(adata.obs['celltype'])

    return temp_adata

if __name__ == "__main__":
    random.seed(3)
    np.random.seed(3)
    adata = sc.read_h5ad(os.path.join(data_dir, 'dataset10_total.h5ad'))
    print(np.unique(np.array(adata.obs['celltype'])))
    slice_adata = adata[(np.array(adata.obs['celltype']) == 'MEP') | (np.array(adata.obs['celltype']) == 'GMP') | (
                np.array(adata.obs['celltype']) == 'CMP')]

    slice_adata.obs['celltype'] = np.array(assign_labels_to_numbers(slice_adata.obs['celltype']))
    # adata.obs['celltype'] = np.array(assign_labels_to_numbers(adata.obs['celltype']))
    print(slice_adata.shape)
    # plot_umap(slice_adata)
    adata1 = slice_adata[slice_adata.obs['batch'] == 1, :].copy()
    adata2 = slice_adata[slice_adata.obs['batch'] == 2, :].copy()
    source, target, model_shrinking = pre_processing(adata1.X, adata2.X, num_epochs=100,
                                                     load_weights_path=dim_reduce_weights_path)
    adata1 = setX(adata1,source)
    adata2 = setX(adata2,target)
    sc.pp.normalize_total(adata1)  # Normalize data
    sc.pp.normalize_total(adata2)
    sc.pp.log1p(adata1)  # Log-transform data
    sc.pp.log1p(adata2)
    adata1.obs.index.name = 'Cell'  # Set the cell index name
    adata1.var.index.name = 'Gene'  # Set the gene index name
    adata2.obs.index.name = 'Cell'  # Set the cell index name
    adata2.var.index.name = 'Gene'  # Set the gene index name
    adata1.uns['sample_name'] = 'Person1Day1'
    adata2.uns['sample_name'] = 'Person1Day2'

    adata_list = [adata1, adata2]
    ifnb_liger = pyliger.create_liger(adata_list, remove_missing=False)

    pyliger.normalize(ifnb_liger,remove_missing=False)
    pyliger.select_genes(ifnb_liger)
    pyliger.scale_not_center(ifnb_liger)

    pyliger.optimize_ALS(ifnb_liger, k=6)
    pyliger.quantile_norm(ifnb_liger)
    pyliger.leiden_cluster(ifnb_liger, resolution=1)
    src_batch, target_batch = ifnb_liger.adata_list

    src_batch = np.array(src_batch.obsm['H'])
    target_batch = np.array(target_batch.obsm['H'])

    adata = sc.AnnData(X=np.concatenate((src_batch, target_batch), axis=0))
    adata.obs['batch'] = np.concatenate((np.array([0] * len(src_batch)), np.array([1] * len(target_batch))))
    adata.obs['celltype'] = np.array(slice_adata.obs["celltype"])
    # sc.tl.pca(adata, svd_solver='arpack', n_comps=5)
    # silhouette_coeff_ASW(adata, method_use='raw', embed='X_pca')

    plot_umap_batch(adata)
    plot_umap_celltype(adata)
