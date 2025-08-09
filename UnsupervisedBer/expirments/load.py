import numpy as np
import scanpy as sc
from sklearn.impute import SimpleImputer


def make_adata_from_batches(batch1, batch2, batch1_labels, batch2_labels):
    adata = sc.AnnData(X=np.concatenate((batch1, batch2), axis=0))
    adata.obs['batch'] = [1 if i < len(batch1) else 2 for i in range(len(batch1) + len(batch2))]
    labels = np.concatenate((batch1_labels, batch2_labels), axis=0)
    adata.obs['celltype'] = labels
    sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
    adata.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat

    return adata


def assign_labels_to_numbers(labels):
    """
    Assigns a unique number to each distinct string label.

    Args:
        labels (list): List of string labels.

    Returns:
        dict: A dictionary mapping each label to its corresponding number.
    """
    unique_labels = set(labels)
    label_to_number = {label: number for number, label in enumerate(unique_labels)}
    number_to_label = {number: label for number, label in enumerate(unique_labels)}
    encoding_labels = [label_to_number[label] for label in labels]

    return encoding_labels, number_to_label


def get_batch_from_adata(adata):
    batch_name = list(set(adata.obs['batch']))
    adata1 = adata[adata.obs['batch'] == batch_name[0], :].copy()
    adata2 = adata[adata.obs['batch'] == batch_name[1], :].copy()

    # adata1 = adata[adata.obs['batch'] == 1]
    # adata2 = adata[adata.obs['batch'] == 2]

    return adata1, adata2


def load_to_adata_shaham_dataset(path_src, path_target, path_src_labels, path_target_labels):
    batch1 = np.loadtxt(path_src, delimiter=',')
    batch2 = np.loadtxt(path_target, delimiter=',')
    zeros_array_b1 = np.zeros((batch1.shape[0], 25))
    zeros_array_b2 = np.zeros((batch2.shape[0], 25))
    batch1 = np.concatenate((batch1, zeros_array_b1), axis=1)
    batch2 = np.concatenate((batch2, zeros_array_b2), axis=1)

    batch1_labels = np.loadtxt(path_src_labels, delimiter=',')
    batch2_labels = np.loadtxt(path_target_labels, delimiter=',')
    adata = make_adata_from_batches(batch1, batch2, batch1_labels, batch2_labels)
    # adata.obs['batch'] = adata.obs['batch'].astype(str).astype("category")
    # adata.obs['celltype'] = adata.obs['celltype'].astype(str).astype("category")
    adata.layers['counts'] = adata.X
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
    sc.pp.neighbors(adata)

    adata.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat

    # sc.pp.scale(adata)

    return adata


def remove_nan(data):
    if np.any(np.isnan(data)):
        my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data = my_imputer.fit_transform(data)

    return data


def pre_process_cytof_data(data):
    return np.log2(1 + data)


def load_and_pre_process_data(src_path, target_path):
    src_data = np.loadtxt(src_path, delimiter=',')
    target_data = np.loadtxt(target_path, delimiter=',')

    src_data = remove_nan(src_data)
    target_data = remove_nan(target_data)

    src_data = pre_process_cytof_data(src_data)
    target_data = pre_process_cytof_data(target_data)

    return src_data, target_data
