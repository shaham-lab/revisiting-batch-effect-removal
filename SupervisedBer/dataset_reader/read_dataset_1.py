import os
from pathlib import Path

import numpy as np
import scanpy as sc
import pandas as pd

from expirements.load import assign_labels_to_numbers
from expirements.utils import plot_adata


def get_dataset():
    data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\BER\new_data\dataset1"
    expr_filename = os.path.join(data_dir, 'dataset1_sm_uc3.txt')

    adata = sc.read_text(expr_filename, delimiter='\t', first_column_names=True, dtype='float64')
    # print(len(adata))
    adata = adata.T
    # Read sample info
    metadata_filename = os.path.join(data_dir, "sample_sm_uc3.txt")
    sample_adata = pd.read_csv(metadata_filename, header=0, index_col=0, sep='\t')

    adata.obs['batch'] = sample_adata.loc[adata.obs_names, "batch"]
    adata.obs['celltype'] = sample_adata.loc[adata.obs_names, "celltype"]
    sc.pp.filter_cells(adata, min_genes=300)
    sc.pp.filter_genes(adata, min_cells=5)
    # sc.pp.normalize_total(adata)
    adata.layers['counts'] = adata.X
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
    sc.pp.neighbors(adata)

    adata.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat
    encoding_labels,number_to_label = assign_labels_to_numbers(adata.obs['batch'])
    adata.obs['batch'] = np.array(encoding_labels)

    adata.obs['batch'] = adata.obs['batch'].astype(str).astype("category")
    adata.obs['celltype'] = adata.obs['celltype'].to_list()
    # adata.obs['celltype'] = adata.obs['celltype'].astype(str).astype("category")
    plot_adata(adata, embed="dim_reduce", title='before')

    return adata


if __name__ == "__main__":
    filepath = r"adata_dataset_1_norm.h5ad"
    filepath = Path(filepath)
    print(f'Write anndata to {filepath}')
    adata = get_dataset()
    # adata.write(filepath)
#     hvgs = adata.var.index
#     adata = scib.preprocessing.hvg_batch(
#         adata,
#         batch_key="batch",
#         target_genes=2000,
#             adataOut=True
#     )
#     adata = scib.preprocessing.scale_batch(adata, "batch")
#
#     plot_adata(adata, embed='X_pca', plot_dir="",
#                title='before')
