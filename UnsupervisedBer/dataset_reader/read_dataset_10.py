from pathlib import Path

import numpy as np
import scanpy as sc
import pandas as pd

import os

from expirments.load import assign_labels_to_numbers

data_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\batch_effect\dataset10"
parent_dir = r"C:\Users\avrah\OneDrive\שולחן העבודה\batch_effect\dataset10"
def make_dataset_10():
    myData1 = pd.read_csv(os.path.join(parent_dir,'b1_exprs.txt/b1_exprs.txt'),header=0, index_col=0, sep='\t')
    myData2 = pd.read_csv(os.path.join(parent_dir,'b2_exprs.txt/b2_exprs.txt'),header=0, index_col=0, sep='\t')
    mySample1 = pd.read_csv(os.path.join(parent_dir,'b1_celltype.txt/b1_celltype.txt'),header=0, index_col=0, sep='\t')
    mySample2 = pd.read_csv(os.path.join(parent_dir,'b2_celltype.txt/b2_celltype.txt'),header=0, index_col=0, sep='\t')

    adata1 = sc.AnnData(myData1.values.T)
    adata1.obs_names = myData1.keys()
    adata1.var_names = myData1.index
    adata1.obs['celltype'] = mySample1.loc[adata1.obs_names,['CellType']]
    adata1.obs['batch'] = 1
    adata1.obs['batch'] = adata1.obs['batch'].astype('category')
    adata1.obs['blb'] = 'batch1'

    adata2 = sc.AnnData(myData2.values.T)
    adata2.obs_names = myData2.keys()
    adata2.var_names = myData2.index
    adata2.obs['celltype'] = mySample2.loc[adata2.obs_names,['CellType']]
    adata2.obs['batch'] = 2
    adata2.obs['batch'] = adata2.obs['batch'].astype('category')
    adata2.obs['blb'] = 'batch2'

    # Combine 2 dataframe to run PCA
    # adata = sc.AnnData(adata1, adata2, batch_key='batch')
    # adata
    adata = sc.AnnData(np.concatenate([adata1.X, adata2.X]))
    adata.obs_names = adata1.obs_names.tolist() + adata2.obs_names.tolist()
    adata.var_names = adata1.var_names.tolist()
    adata.obs['celltype'] = adata1.obs['celltype'].tolist() + adata2.obs['celltype'].tolist()
    adata.obs['batch'] = adata1.obs['batch'].tolist() + adata2.obs['batch'].tolist()
    adata.obs['blb'] = adata1.obs['blb'].tolist() + adata2.obs['blb'].tolist()

    # adata = sc.read_h5ad(os.path.join(parent_dir, 'myTotalData.h5ad'))
    adata.layers['counts'] = adata.X

    sc.pp.filter_cells(adata, min_genes=300)
    sc.pp.filter_genes(adata, min_cells=5)
    # sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    # zero_columns = np.all(adata.X == 0, axis=0)
    # adata.X = adata.X[:, ~zero_columns]

    npcs_train = 20
    sc.tl.pca(adata, svd_solver='arpack', n_comps=npcs_train)  # output save to adata.obsm['X_pca']
    adata.obsm['X_pca'] *= -1
    adata.write_h5ad(os.path.join(parent_dir, 'dataset10_total.h5ad'))


def get_dataset():
    make_dataset_10()
    adata = sc.read_h5ad(os.path.join(data_dir, 'dataset10_total.h5ad'))
    # adata.obs['celltype'] = np.array(assign_labels_to_numbers(adata.obs['celltype']))
    adata.obs['batch'] = adata.obs['batch'].astype(str).astype("category")
    adata.obs['celltype'] = adata.obs['celltype'].astype(str).astype("category")

    return adata


if __name__ == "__main__":
    filepath = r"adata_dataset_10_norm.h5ad"
    filepath = Path(filepath)

    adata = get_dataset()
    adata.write(filepath)