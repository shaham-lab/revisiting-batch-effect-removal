import os

import numpy as np
import scanpy as sc
import pandas as pd
import scib
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

# from expirments.load import assign_labels_to_numbers
# from metrics import calculate_ASW, calculate_ari, compute_lisi_adata, ari_calcul_func_adata


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

    # Identify highly variable genes

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="cell_ranger", batch_key="batch")
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)

    # sc.tl.pca(adata, svd_solver='arpack',n_comps=20)
    adata.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat, same scale as total_ann


    return adata


if __name__ == "__main__":
    adata = get_dataset()
    print(adata.layers["counts"])
    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="celltype",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=["X_pca"],
        n_jobs=1,
    )
    bm.benchmark()

    from rich import print

    df = bm.get_results(min_max_scale=False).transpose().to_csv("result1.csv")



