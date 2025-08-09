import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from torch import nn

from tqdm import tqdm
class MMD(nn.Module):
    def __init__(self,
                 src,
                 target,
                 target_sample_size=1000,
                 n_neighbors=25,
                 scales=None,
                 weights=None):
        super(MMD, self).__init__()
        if scales is None:
            med_list = torch.zeros(25)
            for i in range(25):
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
        # expand dist to a 1xnxm tensor where the 1 is broadcastable
        sQdist = (torch.cdist(X, Y) ** 2).unsqueeze(0)
        scales = self.scales.unsqueeze(-1).unsqueeze(-1)
        weights = self.weights.unsqueeze(-1).unsqueeze(-1)

        return torch.sum(weights * torch.exp(-sQdist / (torch.pow(scales, 2))), 0)

    # Calculate the MMD cost
    def cost(self):
        mmd_list = torch.zeros(25)
        for i in range(25):
            src = self.src[torch.randint(0, self.src.shape[0] - 1, (self.target_sample_size,))]
            target = self.target[torch.randint(0, self.target.shape[0] - 1, (self.target_sample_size,))]
            xx = self.kernel(src, src)
            xy = self.kernel(src, target)
            yy = self.kernel(target, target)
            # calculate the bias MMD estimater (cannot be less than 0)
            MMD = torch.mean(xx) - 2 * torch.mean(xy) + torch.mean(yy)
            mmd_list[i] = torch.sqrt(MMD)

            # return the square root of the MMD because it optimizes better
        return torch.mean(mmd_list)

def silhouette_coeff_ASW(adata, method_use='raw',embed='X_pca', save_dir='', save_fn='', percent_extract=0.8):
    random.seed(0)
    asw_fscore = []
    asw_bn = []
    asw_bn_sub = []
    asw_ctn = []
    iters = []
    if embed=='':
        embed_data= adata.X
    else:
        embed_data= adata.obsm[embed]

    for i in tqdm(range(20)):
        iters.append('iteration_' + str(i + 1))
        rand_cidx = np.random.choice(adata.obs_names, size=int(len(adata.obs_names) * percent_extract), replace=False)
        #         print('nb extracted cells: ',len(rand_cidx))
        adata_ext = adata[rand_cidx, :]
        if embed == '':
            embed_data = adata_ext.X
        else:
            embed_data = adata_ext.obsm[embed]

        asw_batch = silhouette_score(embed_data, adata_ext.obs['batch'])
        asw_celltype = silhouette_score(embed_data, adata_ext.obs['celltype'])
        min_val = -1
        max_val = 1
        asw_batch_norm = (asw_batch - min_val) / (max_val - min_val)
        asw_celltype_norm = (asw_celltype - min_val) / (max_val - min_val)

        fscoreASW = (2 * (1 - asw_batch_norm) * (asw_celltype_norm)) / (1 - asw_batch_norm + asw_celltype_norm)
        asw_fscore.append(fscoreASW)
        asw_bn.append(asw_batch_norm)
        asw_bn_sub.append(1 - asw_batch_norm)
        asw_ctn.append(asw_celltype_norm)

    #     iters.append('median_value')
    #     asw_fscore.append(np.round(np.median(fscoreASW),3))
    #     asw_bn.append(np.round(np.median(asw_batch_norm),3))
    #     asw_bn_sub.append(np.round(1 - np.median(asw_batch_norm),3))
    #     asw_ctn.append(np.round(np.median(asw_celltype_norm),3))
    df = pd.DataFrame({ 'asw_batch_norm_sub_mean': [np.array(asw_bn_sub).mean()], 'asw_batch_norm_sub_std': [np.array(asw_bn_sub).std()],
                       'asw_celltype_norm_mean': [np.array(asw_ctn).mean()],'asw_celltype_norm_std': [np.array(asw_ctn).std()],
                        'fscore_mean': [np.array(asw_fscore).mean()],'fscore_std': [np.array(asw_fscore).std()]
                       })
    print(df)
    print('Save output of pca in: ', save_dir)
    print(df.values.shape)
    print(df.keys())
    return df


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import random


def calculate_ari(myData, cpcs, method_use='resnet', celltypelb='celltype', batchlb='batch'):
    """
    Perform subsampling, clustering, and calculate Adjusted Rand Index (ARI) scores.

    Parameters:
    - myData: DataFrame containing PCs, batch, and cell type columns.
    - cpcs: List of column indices for the PCs in myData.
    - method_use: Name of the batch correction method (default: 'resnet').
    - celltypelb: Column name for cell type (default: 'celltype').
    - batchlb: Column name for batch (default: 'batch').

    Returns:
    - DataFrame of ARI scores.
    """
    np.random.seed(0)
    random.seed(0)

    # Unique cell types and their count
    cell_types = myData[celltypelb].unique()
    num_cell_types = len(cell_types)

    # Parameters
    n_iterations = 20
    subsample_percent = 0.8

    # Lists to store ARI scores
    ari_batch_scores = []
    ari_celltype_scores = []

    for _ in range(n_iterations):
        # Subsampling: extract 80% of cells for each cell type
        selected_indices = []
        for cell_type in cell_types:
            cell_indices = myData[myData[celltypelb] == cell_type].index.tolist()
            sample_size = round(len(cell_indices) * subsample_percent)
            selected_indices.extend(random.sample(cell_indices, sample_size))

        # Create subsampled dataset
        sampled_data = myData.loc[selected_indices]

        # Clustering using KMeans
        kmeans = KMeans(n_clusters=num_cell_types, random_state=0)
        sampled_data['cluster'] = kmeans.fit_predict(sampled_data.iloc[:, cpcs])

        # Calculate ARI scores
        ari_batch = adjusted_rand_score(sampled_data[batchlb], sampled_data['cluster'])
        ari_celltype = adjusted_rand_score(sampled_data[celltypelb], sampled_data['cluster'])

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


# Example usage:
# Assume `myData` is a pandas DataFrame containing your data with columns for PCs, batch, and cell type.
# Assume `cpcs` is a list of column indices for the principal components.
# results = calculate_ari(myData, cpcs)

def eval_mmd(source, target):
    mmd_value = MMD(source, target).cost()

    return mmd_value