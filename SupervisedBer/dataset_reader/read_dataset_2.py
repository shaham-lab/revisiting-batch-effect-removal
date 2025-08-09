import os
from pathlib import Path

import numpy as np
import scanpy as sc
import pandas as pd

from expirements.load import assign_labels_to_numbers
from expirements.utils import plot_adata


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

    encoding_labels,number_to_label = assign_labels_to_numbers(adata.obs['batch'])
    adata.obs['batch'] = np.array(encoding_labels)

    adata.obs['batch'] = adata.obs['batch'].astype(str).astype("category")
    adata.obs['celltype'] = adata.obs['celltype'].to_list()
    # adata.obs['celltype'] = adata.obs['celltype'].astype(str).astype("category")
    plot_adata(adata, embed="dim_reduce", title='before')

    return adata


if __name__ == "__main__":
    filepath = r"adata_dataset_2_norm.h5ad"
    filepath = Path(filepath)
    print(f'Write anndata to {filepath}')
    adata = get_dataset()
    adata.write(filepath)
