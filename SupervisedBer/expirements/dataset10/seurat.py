# Import the necessary packages
import os
from pathlib import Path

import scanpy as sc
import numpy as np

import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import normalize

# from scDML.scDML.metrics import evaluate_dataset

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
from expirements.utils import make_combinations_from_config, sample_from_space, plot_adata
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
    "save_weights": os.path.join(parent_dir, "weights/seuratv3/dataset10-benchmark/"),
    "plots_dir": os.path.join(parent_dir, "plots/seuratv3/dataset10-benchmark/")
}
dim_reduce_weights_path = os.path.join(config["save_weights"], "dim_reduce")
os.makedirs(dim_reduce_weights_path, exist_ok=True)
os.makedirs(config["plots_dir"], exist_ok=True)

# configurations = make_combinations_from_config(config)
# configurations = sample_from_space(configurations, num_of_samples=3)


# configurations = [configurations[1]]
def setX(adata, x):
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
    encoding_labels,number_to_label = assign_labels_to_numbers(slice_adata.obs['celltype'])

    slice_adata.obs['celltype'] = np.array(encoding_labels)
    # adata.obs['celltype'] = np.array(assign_labels_to_numbers(adata.obs['celltype']))
    print(slice_adata.shape)
    # plot_umap(slice_adata)
    adata1 = slice_adata[slice_adata.obs['batch'] == 1, :].copy()
    adata2 = slice_adata[slice_adata.obs['batch'] == 2, :].copy()
    source, target, model_shrinking = pre_processing(adata1.X, adata2.X, num_epochs=100,
                                                     load_weights_path=dim_reduce_weights_path)
    adata1 = setX(adata1, source)
    adata2 = setX(adata2, target)
    # sc.pp.normalize_total(adata1)  # Normalize data
    # sc.pp.normalize_total(adata2)
    # sc.pp.log1p(adata1)  # Log-transform data
    # sc.pp.log1p(adata2)

    import anndata as ad

    t_adata = ad.concat([adata1, adata2],index_unique="-", join="inner")
    t_adata.obs.index.name = 'Cell'  # Set the cell index name
    t_adata.var.index.name = 'Gene'  # Set the gene index name

    scaler = MinMaxScaler()
    t_adata.X = scaler.fit_transform(t_adata.X)
    sc.pp.normalize_total(t_adata)

    sc.pp.filter_cells(t_adata, min_genes=1)
    silhouette_coeff_ASW(t_adata,embed='').to_csv(os.path.join(config["plots_dir"],
                                                      "before.csv"))

    # Identify highly variable genes using the Seurat flavor
    sc.pp.highly_variable_genes(t_adata, batch_key='batch', flavor='seurat_v3', n_top_genes=3000)
    # Regress out the batch effect
    sc.pp.regress_out(t_adata, ['batch'])
    plot_adata(t_adata, embed='X_pca', plot_dir=config["plots_dir"],
               title='after-calibration-seurat')

    silhouette_coeff_ASW(t_adata,embed='').to_csv(os.path.join(config["plots_dir"],
                                                      "calibrated_adata_seurat.csv"))

    # t_adata.X = scaler.inverse_transform(t_adata.X)
    # plot_adata(t_adata)
    # adata1, adata2 = get_batch_from_adata(t_adata)
    #
    # print(eval_mmd(torch.tensor(adata1.X), torch.tensor(adata2.X)))
