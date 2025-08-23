import random

import pandas as pd
import sklearn
import torch
from sklearn.cluster import KMeans
from torch import nn
from tqdm import tqdm

class MMD(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) implementation for measuring distribution differences.
    
    MMD is a kernel-based method for measuring the difference between two probability
    distributions. It uses a kernel function to map data to a high-dimensional
    feature space where the difference between distributions can be measured
    using the mean embeddings.
    
    Attributes:
        src (torch.Tensor): Source distribution samples
        target (torch.Tensor): Target distribution samples
        target_sample_size (int): Number of samples to use for scale calculation
        n_neighbors (int): Number of nearest neighbors for scale calculation
        scales (torch.Tensor): Kernel bandwidth scales
        weights (torch.Tensor): Weights for different kernel scales
        kernel (function): Kernel function to use
    """
    
    def __init__(self,
                 src,
                 target,
                 target_sample_size=1000,
                 n_neighbors=500,
                 scales=None,
                 weights=None):
        """
        Initialize the MMD module.
        
        Args:
            src (torch.Tensor): Source distribution samples
            target (torch.Tensor): Target distribution samples
            target_sample_size (int): Number of samples to use for scale calculation
            n_neighbors (int): Number of nearest neighbors for scale calculation
            scales (torch.Tensor, optional): Pre-computed kernel scales
            weights (torch.Tensor, optional): Weights for different kernel scales
        """
        super(MMD, self).__init__()
        if scales is None:
            med_list = torch.zeros(5)
            for i in range(5):
                sample = target[torch.randint(0, target.shape[0] - 1, (target_sample_size,))]
                distance_matrix = torch.cdist(sample, sample)
                sorted, indices = torch.sort(distance_matrix, dim=0)

                # nearest neighbor is the point so we need to exclude it
                med_list[i] = torch.median(sorted[:, 1:n_neighbors])
            med = torch.mean(med_list)

        scales = [med / 2, med, med * 2]  # CyTOF

        # print(scales)
        scales = torch.tensor(scales)
        weights = torch.ones(len(scales))
        self.src = src
        self.target = target
        self.target_sample_size = target_sample_size
        self.kernel = self.RaphyKernel
        self.scales = scales
        self.weights = weights

    def RaphyKernel(self, X, Y):
        """
        Compute Raphy kernel between two sets of points.
        
        The Raphy kernel is a multi-scale Gaussian kernel that uses multiple
        bandwidths to capture different scales of structure in the data.
        
        Args:
            X (torch.Tensor): First set of points
            Y (torch.Tensor): Second set of points
        
        Returns:
            torch.Tensor: Kernel matrix between X and Y
        """
        # expand dist to a 1xnxm tensor where the 1 is broadcastable
        sQdist = (torch.cdist(X, Y) ** 2).unsqueeze(0)
        scales = self.scales.unsqueeze(-1).unsqueeze(-1)
        weights = self.weights.unsqueeze(-1).unsqueeze(-1)

        return torch.sum(weights * torch.exp(-sQdist / (torch.pow(scales, 2))), 0)

    # Calculate the MMD cost
    def cost(self):
        """
        Calculate the MMD cost between source and target distributions.
        
        This function computes the MMD estimate by sampling from both distributions
        and computing the kernel-based distance. It performs multiple trials to
        get a stable estimate.
        
        Returns:
            tuple: (mmd_mean, mmd_std)
                - mmd_mean (torch.Tensor): Mean MMD estimate
                - mmd_std (torch.Tensor): Standard deviation of MMD estimates
        """
        mmd_list = torch.zeros(75)
        for i in range(75):
            src = self.src[torch.randint(0, self.src.shape[0] - 1, (self.target_sample_size,))]
            target = self.target[torch.randint(0, self.target.shape[0] - 1, (self.target_sample_size,))]
            xx = self.kernel(src, src)
            xy = self.kernel(src, target)
            yy = self.kernel(target, target)
            # calculate the bias MMD estimater (cannot be less than 0)
            MMD = torch.mean(xx) - 2 * torch.mean(xy) + torch.mean(yy)
            mmd_list[i] = torch.sqrt(MMD)
        mmd_mean = torch.mean(mmd_list)
        mmd_std= torch.std(mmd_list)
            # return the square root of the MMD because it optimizes better
        return mmd_mean,mmd_std


def eval_mmd(source, target):
    """
    Evaluate MMD between source and target distributions.
    
    This function is a convenience wrapper that creates an MMD object and
    computes the MMD cost between two distributions. It handles dimension
    matching by truncating to the minimum dimension.
    
    Args:
        source (torch.Tensor): Source distribution samples
        target (torch.Tensor): Target distribution samples
    
    Returns:
        tuple: (mmd_mean, mmd_std) - Mean and standard deviation of MMD estimates
    """
    len_source = min(source.shape[1],target.shape[1])
    s1 =source[:,:len_source]
    s2 =target[:,:len_source]
    mmd_mean,mmd_std = MMD(s1, s2).cost()

    return mmd_mean,mmd_std


import numpy as np
from sklearn.metrics import pairwise_distances, adjusted_rand_score, pair_confusion_matrix

from sklearn.metrics import silhouette_score


def silhouette(adata, group_key='batch', metric='euclidean', scale=True):
    """
    wrapper for sklearn silhouette function values range from [-1, 1] with 1 being an ideal fit, 0 indicating
    overlapping clusters and -1 indicating misclassified cells
    :param group_key: key in adata.obs of cell labels
    :param embed: embedding key in adata.obsm, default: 'X_pca'
    """
    asw = silhouette_score(
        X=adata.X,
        labels=adata.obs[group_key],
        metric=metric
    )
    if scale:
        asw = (asw + 1) / 2
    return asw


import torch
import numpy as np


def batch_kl(adata, batch_key='batch'):
    """
    Calculate the BatchKL metric for batch correction.

    Args:
        meta_data (numpy.ndarray): Meta data (e.g., batch information).
        embedding (torch.Tensor): Low-dimensional embeddings (e.g., PCA, t-SNE, UMAP).

    Returns:
        float: BatchKL metric.
    """
    # Assuming meta_data contains batch information (e.g., batch labels)
    # and embedding is a torch.Tensor with shape (n_cells, n_features)

    # Calculate mean and variance for each batch
    batch_means = []
    batch_vars = []
    adata1 = adata[adata.obs[batch_key] == 1, :].copy()
    adata2 = adata[adata.obs[batch_key] == 2, :].copy()

    unique_batches = np.unique(adata.obs[batch_key])
    for batch in unique_batches:
        batch_indices = np.where(adata.obs[batch_key] == batch)[0]
        batch_embedding = adata[batch_indices].X
        batch_mean = np.mean(batch_embedding, axis=0)
        batch_var = np.var(batch_embedding, axis=0)
        batch_means.append(batch_mean)
        batch_vars.append(batch_var)

    # Compute BatchKL
    batch_kl_sum = 0.0
    for i in range(len(unique_batches)):
        for j in range(i + 1, len(unique_batches)):
            kl_divergence = np.sum(
                0.5 * (batch_vars[i] / batch_vars[j] + (batch_means[j] - batch_means[i]) ** 2 / batch_vars[
                    j] - 1.0 + np.log(batch_vars[j] / batch_vars[i]))
            )
            batch_kl_sum += kl_divergence

    # Normalize by the number of batch pairs
    num_batch_pairs = len(unique_batches) * (len(unique_batches) - 1) // 2
    batch_kl = batch_kl_sum / num_batch_pairs

    return batch_kl


def compute_kbet(adata, batch_key='batch'):
    """
    Compute the kBET score for batch effect assessment.

    Args:
        adata (AnnData): Annotated data matrix.
        batch_key (str): Column name in adata.obs containing batch information.

    Returns:
        float: kBET score.
    """
    # Extract batch labels and cell embeddings
    batch_labels = adata.obs[batch_key]
    cell_embeddings = adata.X

    # Compute pairwise distances between cells
    dist_matrix = pairwise_distances(cell_embeddings, metric='euclidean')

    # Calculate average distance within each batch
    intra_batch_distances = []
    for batch_id in np.unique(batch_labels):
        batch_indices = np.where(batch_labels == batch_id)[0]
        batch_distances = dist_matrix[batch_indices][:, batch_indices]
        intra_batch_distances.append(np.mean(batch_distances))

    # Compute average distance between batches
    inter_batch_distances = np.mean(intra_batch_distances)

    # Calculate kBET score
    k_bet_score = inter_batch_distances / np.mean(intra_batch_distances)

    return k_bet_score


import scanpy as sc


# Find optimal resolution given ncluster
def find_resolution(adata_, n_clusters, random):
    adata = adata_.copy()
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions) / 2
        sc.tl.louvain(adata, resolution=current_res, random_state=random)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))

        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
        iteration = iteration + 1
    return current_res
import numpy as np
from scipy.spatial.kdtree import KDTree


def compute_lisi(X, meta_data, label_colnames, perplexity=30, nn_eps=0):
    """
    Compute Local Inverse Simpson's Index (LISI) scores.

    Args:
        X (np.ndarray): Matrix with cells (rows) and features (columns).
        meta_data (pandas.DataFrame): Data frame with one row per cell.
        label_colnames (list): List of variable names to compute LISI for.
        perplexity (int, optional): Effective number of each cell's neighbors. Defaults to 30.
        nn_eps (float, optional): Error bound for nearest neighbor search. Defaults to 0.

    Returns:
        pandas.DataFrame: Data frame of LISI values. Each row is a cell and each
                         column is a different label variable.

    Raises:
        ValueError: If any label contains missing values.
    """

    N = X.shape[0]
    k = perplexity * 3  # Number of neighbors to search

    # Find nearest neighbors using KDTree
    tree = KDTree(X)
    distances, indices = tree.query(X, k=k + 1, eps=nn_eps)  # Include itself for exclusion later

    lisi_df = np.full((N, len(label_colnames)), np.nan)
    for i, label_colname in enumerate(label_colnames):
        labels = meta_data[label_colname].values

        # Check for missing values
        if np.isnan(labels).any():
            raise ValueError(f"Cannot compute LISI on missing values in '{label_colname}'")

        # Convert labels to integers and exclude self-index
        labels = labels.astype(int) - 1
        indices = indices[:, 1:] - 1  # Exclude first neighbor (self)

        # Calculate Simpson index
        simpson = compute_simpson_index(distances[:, 1:], indices, labels, len(np.unique(labels)), perplexity)
        lisi_df[:, i] = 1 / simpson

    lisi_df = pd.DataFrame(lisi_df, columns=label_colnames)
    lisi_df.index = meta_data.index
    return lisi_df

def compute_simpson_index(distances, neighbor_indices, labels, num_classes, perplexity):
    """
    Compute the Simpson Index for each cell in the neighborhood, weighted by distances.

    Args:
        distances (np.ndarray): Distances to nearest neighbors (rows = cells, columns = neighbors).
        neighbor_indices (np.ndarray): Indices of neighbors for each cell.
        labels (np.ndarray): Array of class labels for each cell.
        num_classes (int): Total number of unique labels.
        perplexity (int): Perplexity value to determine the effective neighborhood size.

    Returns:
        np.ndarray: Simpson Index for each cell.
    """
    N = neighbor_indices.shape[0]
    simpson_indices = np.zeros(N)

    # Scale distances into weights using a Gaussian kernel
    sigma = np.mean(distances)  # Bandwidth parameter
    weights = np.exp(-distances ** 2 / (2 * sigma ** 2))

    for i in range(N):
        # Get the labels of the neighbors
        neighbor_labels = labels[neighbor_indices[i]]

        # Weight counts by distances
        weighted_counts = np.zeros(num_classes)
        for j, label in enumerate(neighbor_labels):
            weighted_counts[label] += weights[i, j]

        # Normalize to compute proportions
        proportions = weighted_counts / np.sum(weighted_counts)

        # Compute the Simpson Index for this cell
        simpson_indices[i] = np.sum(proportions ** 2)

    return simpson_indices

import os
import numpy as np
import pandas as pd


def normalize_values(adata, colns, min_val=1, max_val=2):
    """
    Normalize median values of specified numeric columns in an AnnData object.
    """
    med_val = []
    med_val_norm = []
    methods_use = []

    # Loop through numeric columns in the `obs` of AnnData
    for col in adata.obs.columns:
        if np.issubdtype(adata.obs[col].dtype, np.number):
            methods_use.append(col)
            median_val = np.median(adata.obs[col])
            med_val.append(median_val)
            med_val_norm.append((median_val - min_val) / (max_val - min_val))

    # Create a DataFrame with normalized values
    myDataNorm = pd.DataFrame({
        colns[0]: methods_use,
        colns[1]: med_val,
        colns[2]: med_val_norm
    })

    return myDataNorm


def get_celltype_common(adata):
    """
    Get common cell types across batches in an AnnData object.
    """
    # Ensure 'batch' and 'cell_type' are present in `adata.obs`
    if 'batch' not in adata.obs or 'cell_type' not in adata.obs:
        raise KeyError("The AnnData object must have 'batch' and 'cell_type' in .obs.")

    batches = adata.obs['batch'].unique()
    celltypels = adata.obs['cell_type'].unique()
    print(f"Unique cell types: {celltypels}")

    if len(batches) > 1:
        ctls = []

        # Group by batch and collect unique cell types
        for b in batches:
            ct = adata.obs.loc[adata.obs['batch'] == b, 'cell_type'].unique()
            ctls.append(ct)
            print(f"Batch: {b}")
            print(f"Cell types: {ct}")

        # Find common cell types across batches
        ct_common = set(ctls[0])
        for ct in ctls[1:]:
            ct_common = ct_common.intersection(ct)

        ct_common = list(ct_common)
        print(f"Common cell types: {ct_common}")

        cells_common = adata.obs.index[adata.obs['cell_type'].isin(ct_common)]
        return {
            'ct_common': ct_common,
            'cells_common': cells_common,
            'batches': batches,
            'celltypels': celltypels
        }
    else:
        return None


def run_LISI_final(adata, save_dir, eval_metric, methods_use, plx=40):
    """
    Run LISI analysis on an AnnData object.
    """
    # Ensure embeddings are present in `adata.obsm`
    if 'X_pca' not in adata.obsm:
        raise KeyError("The AnnData object must have PCA embeddings in .obsm['X_pca'].")

    # Extract the first 20 components of PCA
    lisi_embeddings = adata.obsm['X_pca'][:, :20]

    # Ensure 'batch' and 'cell_type' are in `adata.obs`
    if 'batch' not in adata.obs or 'cell_type' not in adata.obs:
        raise KeyError("The AnnData object must have 'batch' and 'cell_type' in .obs.")

    lisi_meta_data = adata.obs[['batch', 'cell_type']]
    lisi_labels = ['batch', 'cell_type']

    # Compute LISI
    lisi_res = compute_lisi(lisi_embeddings, lisi_meta_data, lisi_labels, perplexity=plx)
    lisi_res['cell'] = adata.obs_names

    # Split LISI results for batch and cell type
    lisi_batch = lisi_res[['batch', 'cell']]
    lisi_celltype = lisi_res[['cell_type', 'cell']]

    output_dir = os.path.join(save_dir, eval_metric, methods_use)
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    lisi_batch.to_csv(os.path.join(output_dir, f'lisi_batch_{plx}.txt'), sep='\t', index=False)
    lisi_celltype.to_csv(os.path.join(output_dir, f'lisi_celltype_{plx}.txt'), sep='\t', index=False)


def calculate_ari(adata, cpcs, method_use='resnet', celltypelb='celltype', batchlb='batch'):
    """
    Perform subsampling, clustering, and calculate Adjusted Rand Index (ARI) scores using Scanpy's AnnData.

    Parameters:
    - adata: AnnData object containing PCs, batch, and cell type annotations.
    - cpcs: List of column indices for the PCs in `adata.obsm['X_pca']`.
    - method_use: Name of the batch correction method (default: 'resnet').
    - celltypelb: Key in `adata.obs` for cell type labels (default: 'celltype').
    - batchlb: Key in `adata.obs` for batch labels (default: 'batch').

    Returns:
    - DataFrame of ARI scores.
    """
    import numpy as np
    import random
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    np.random.seed(0)
    random.seed(0)

    # Unique cell types and their count
    cell_types = adata.obs[celltypelb].unique()
    num_cell_types = len(cell_types)

    # Parameters
    n_iterations = 100
    subsample_percent = 0.8

    # Lists to store ARI scores
    ari_batch_scores = []
    ari_celltype_scores = []

    for _ in range(n_iterations):
        # Subsampling: extract 80% of cells for each cell type
        selected_indices = []
        for cell_type in cell_types:
            cell_indices = adata.obs[adata.obs[celltypelb] == cell_type].index.tolist()
            sample_size = round(len(cell_indices) * subsample_percent)
            selected_indices.extend(random.sample(cell_indices, sample_size))

        # Create subsampled AnnData object
        sampled_adata = adata[selected_indices, :]

        # Clustering using KMeans
        kmeans = KMeans(n_clusters=num_cell_types, random_state=0)
        sampled_adata.obs['cluster'] = kmeans.fit_predict(sampled_adata.obsm['X_pca'][:, cpcs])

        # Calculate ARI scores
        ari_batch = adjusted_rand_score(sampled_adata.obs[batchlb], sampled_adata.obs['cluster'])
        ari_celltype = adjusted_rand_score(sampled_adata.obs[celltypelb], sampled_adata.obs['cluster'])

        # Store results
        ari_batch_scores.append(ari_batch)
        ari_celltype_scores.append(ari_celltype)

    # Calculate median ARI scores
    median_ari_batch = np.median(ari_batch_scores)
    median_ari_celltype = np.median(ari_celltype_scores)

    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'method': [method_use] * n_iterations + [f"{method_use}_median"],
        'iteration': list(range(1, n_iterations + 1)) + ['median'],
        'ari_batch': ari_batch_scores + [median_ari_batch],
        'ari_celltype': ari_celltype_scores + [median_ari_celltype]
    })

    # Save results to a text file
    output_file = f"{method_use}_ARI_results.txt"
    results_df.to_csv(output_file, sep='\t', index=False)
    print(f'Results saved to {output_file}')




    return results_df

def calculate_ASW(adata, method_use='raw', save_dir='', save_fn='', percent_extract=0.8):
    random.seed(0)
    asw_fscore = []
    asw_bn = []
    asw_bn_sub = []
    asw_ctn = []
    asw_batch_ls = []
    asw_celltype_ls = []
    iters = []

    for i in tqdm(range(100)):
        iters.append('iteration_' + str(i + 1))
        rand_cidx = np.random.choice(adata.obs_names, size=int(len(adata.obs_names) * percent_extract), replace=False)
        #         print('nb extracted cells: ',len(rand_cidx))
        adata_ext = adata[rand_cidx, :]
        asw_batch = silhouette_score(adata_ext.obsm['X_pca'][:, :20], adata_ext.obs['batch'])
        asw_celltype = silhouette_score(adata_ext.obsm['X_pca'][:, :20], adata_ext.obs['celltype'])
        min_val = -1
        max_val = 1
        asw_batch_norm = (asw_batch - min_val) / (max_val - min_val)
        asw_celltype_norm = (asw_celltype - min_val) / (max_val - min_val)

        fscoreASW = (2 * (1 - asw_batch_norm)*(asw_celltype_norm))/(1 - asw_batch_norm + asw_celltype_norm)
        asw_fscore.append(fscoreASW)
        asw_bn.append(asw_batch_norm)
        asw_bn_sub.append(1 - asw_batch_norm)
        asw_ctn.append(asw_celltype_norm)
        asw_batch_ls.append(asw_batch)
        asw_celltype_ls.append(asw_celltype)
    #     iters.append('median_value')
    #     asw_fscore.append(np.round(np.median(fscoreASW),3))
    #     asw_bn.append(np.round(np.median(asw_batch_norm),3))
    #     asw_bn_sub.append(np.round(1 - np.median(asw_batch_norm),3))
    #     asw_ctn.append(np.round(np.median(asw_celltype_norm),3))
    # df = pd.DataFrame({ 'asw_batch_norm_sub_mean': [np.array(asw_bn_sub).mean()], 'asw_batch_norm_sub_std': [np.array(asw_bn_sub).std()],
    #                    'asw_celltype_norm_mean': [np.array(asw_ctn).mean()],'asw_celltype_norm_std': [np.array(asw_ctn).std()],
    #                     'fscore_mean': [np.array(asw_fscore).mean()],'fscore_std': [np.array(asw_fscore).std()]
    #                    })
    df = pd.DataFrame({'asw_batch_norm':asw_bn, 'asw_batch_norm_sub': asw_bn_sub,
                       'asw_celltype_norm': asw_ctn, 'fscore':asw_fscore,'asw_batch':asw_batch_ls,'asw_celltype':asw_celltype_ls,
                       'method_use':np.repeat(method_use, len(asw_fscore))})
    print(np.median(asw_batch_ls))
    print(np.median(asw_celltype_ls))
    # print(df)
    # print('Save output of pca in: ', save_dir)
    # print(df.values.shape)
    # print(df.keys())
    return df


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from anndata import AnnData




import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

def ari_calcul_func_adata(
    adata,
    is_optimal=False,
    method_use='resnet',
    prefix_coln='X_pca',
    base_name='',
    max_cls=10,
    max_iter=30
):
    # Extract PCA components from adata.obsm
    pcs = range(1, 21)
    cpcs = [f"{prefix_coln}{pc}" for pc in pcs]

    if prefix_coln in adata.obsm:
        data_pca = pd.DataFrame(adata.obsm[prefix_coln], index=adata.obs_names, columns=cpcs)
    else:
        raise ValueError(f"PCA data with prefix '{prefix_coln}' not found in adata.obsm.")

    # Ensure required columns exist in adata.obs
    required_columns = ['batch', 'celltype']
    if not all(col in adata.obs.columns for col in required_columns):
        raise ValueError(f"Missing required columns {required_columns} in adata.obs.")

    # Prepare metadata
    cells_ls = adata.obs_names
    ori_nbcells = len(cells_ls)
    nbct = adata.obs['celltype'].nunique()
    percent_extract = 0.8

    it = []
    total_ari_batch = []
    total_ari_batch_norm = []
    total_ari_celltype = []
    total_ari_celltype_norm = []
    total_fscoreARI = []
    nbiters = 30

    for i in range(1, nbiters + 1):
        # Random sampling of cells
        cells_extract = np.random.choice(cells_ls, size=int(ori_nbcells * percent_extract), replace=False)
        my_pca_ext = data_pca.loc[cells_extract]
        my_obs_ext = adata.obs.loc[cells_extract]

        # Clustering
        if not is_optimal:
            clustering_result = KMeans(n_clusters=nbct, max_iter=max_iter).fit(my_pca_ext)
        else:
            # Implement optimal clustering here
            raise NotImplementedError("Optimal clustering logic is not implemented yet.")

        # Assign cluster labels
        my_obs_ext['clusterlb'] = clustering_result.labels_

        # ARI calculations
        ari_batch = adjusted_rand_score(my_obs_ext['batch'], my_obs_ext['clusterlb'])
        ari_celltype = adjusted_rand_score(my_obs_ext['celltype'], my_obs_ext['clusterlb'])

        # Normalize ARI
        ari_batch_norm = (ari_batch + 1) / 2
        ari_celltype_norm = (ari_celltype + 1) / 2

        # F-score ARI
        fscore_ari = (2 * (1 - ari_batch_norm) * ari_celltype_norm) / (1 - ari_batch_norm + ari_celltype_norm)

        # Collect metrics
        it.append(i)
        total_ari_batch.append(ari_batch)
        total_ari_batch_norm.append(ari_batch_norm)
        total_ari_celltype.append(ari_celltype)
        total_ari_celltype_norm.append(ari_celltype_norm)
        total_fscoreARI.append(fscore_ari)

    # Append median values
    it.append(nbiters + 1)
    total_ari_batch.append(round(np.median(total_ari_batch), 3))
    total_ari_batch_norm.append(round(np.median(total_ari_batch_norm), 3))
    total_ari_celltype.append(round(np.median(total_ari_celltype), 3))
    total_ari_celltype_norm.append(round(np.median(total_ari_celltype_norm), 3))
    total_fscoreARI.append(round(np.median(total_fscoreARI), 3))

    methods = [method_use] * nbiters
    methods.append(f"{method_use}_median")

    # Compile results into a DataFrame
    my_ari = pd.DataFrame({
        "use_case": methods,
        "iteration": it,
        "ari_batch": total_ari_batch,
        "ari_celltype": total_ari_celltype,
        "ari_batch_norm": total_ari_batch_norm,
        "ari_celltype_norm": total_ari_celltype_norm,
        "fscoreARI": total_fscoreARI
    })

    # Write to file
    my_ari.to_csv(f"{base_name}{method_use}_ARI.txt", sep="\t", index=False)

    return my_ari

def kBET(data, batch_labels, k):
    n_neighbors = k
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(data)
    distances, indices = knn.kneighbors(data)

    rejection_rates = []
    for i in range(data.shape[0]):
        batch_counts = np.bincount(batch_labels[indices[i]], minlength=len(np.unique(batch_labels)))
        expected_counts = np.mean(batch_counts)
        observed_counts = batch_counts[batch_labels[i]]
        rejection_rate = 1 - (observed_counts / expected_counts)
        rejection_rates.append(rejection_rate)

    return np.median(rejection_rates)


def summary_KBET(data, batch_labels, k_values):
    results = {}
    for k in k_values:
        results[k] = kBET(data, batch_labels, k)
    return results


def ari(labels_true, labels_pred):
    '''safer implementation of ari score calculation'''
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    tn = int(tn)
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))



