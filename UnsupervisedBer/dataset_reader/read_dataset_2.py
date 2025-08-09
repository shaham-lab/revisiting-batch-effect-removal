import os
from pathlib import Path

import numpy as np
import scanpy as sc
import pandas as pd


def get_dataset():
    data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\BER\new_data\dataset-2"
    expr_filename = os.path.join(data_dir, 'filtered_total_batch1_seqwell_batch2_10x.txt')

    adata = sc.read_text(expr_filename, delimiter='\t', first_column_names=True, dtype='float64')
    adata = adata.T

    # Read sample info
    metadata_filename = os.path.join(data_dir, "filtered_total_sample_ext_organ_celltype_batch.txt")
    sample_adata = pd.read_csv(metadata_filename, header=0, index_col=0, sep='\t')

    adata.obs['batch'] = sample_adata.loc[adata.obs_names, "batch"]
    adata.obs['celltype'] = sample_adata.loc[adata.obs_names, "orig.ident"]

    sc.pp.filter_cells(adata, min_genes=300)
    sc.pp.filter_cells(adata, min_counts=5)

    adata.layers['counts'] = adata.X
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
    sc.pp.neighbors(adata)

    adata.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat

    adata.obs['batch'] = adata.obs['batch'].astype(str).astype("category")
    adata.obs['celltype'] = adata.obs['celltype'].astype(str).astype("category")


    return adata


if __name__ == "__main__":
    filepath = r"adata_dataset_2_norm.h5ad"
    filepath = Path(filepath)

    adata = get_dataset()
    adata.write(filepath)
