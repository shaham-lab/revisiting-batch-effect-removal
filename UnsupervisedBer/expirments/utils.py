import itertools
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from expirments.load import get_batch_from_adata
from plot import get_pca_data, plot_scatter
import scanpy as sc


import umap
import numpy as np

from sklearn.decomposition import PCA

def apply_pca(data):
    """
    Apply PCA to reduce the dimensionality of the data to 2D space.

    Parameters:
    data (numpy.ndarray): The input data to be reduced. Each row represents a data point.

    Returns:
    numpy.ndarray: The data transformed into 2D space.
    """
    # Initialize PCA with 2 components
    pca = PCA(n_components=10)

    # Fit and transform the data
    embedding = pca.fit_transform(data)

    return embedding
def apply_umap(data):
    """
    Apply UMAP to reduce the dimensionality of the data to 2D space.

    Parameters:
    data (numpy.ndarray): The input data to be reduced. Each row represents a data point.

    Returns:
    numpy.ndarray: The data transformed into 2D space.
    """
    # Initialize UMAP with 2 components
    reducer = umap.UMAP(n_components=2)

    # Fit and transform the data
    embedding = reducer.fit_transform(data)

    return embedding


# Example usage:
# Assuming 'data' is a numpy array with your input data
# data = np.random.rand(100, 10)  # Example data
# transformed_data = apply_umap(data)
# print(transformed_data)

def plot_umap_celltype(adata, path, title,celltype_coulmn="celltype"):
    num_pcs = 20
    sc.pp.neighbors(adata, n_pcs=num_pcs, n_neighbors=20)
    sc.tl.umap(adata)
    adata.obs["celltype-label"] = [f"celltype_{id}" for id in adata.obs["celltype"]]
    sc.pl.umap(adata, color='celltype-label')
    import matplotlib.pyplot as plt
    plt.legend(title='')
    plt.savefig(os.path.join(path, f'{title}_celltype.png'))


def plot_umap_batch(adata, path, title):
    num_pcs = 20
    sc.pp.neighbors(adata, n_pcs=num_pcs, n_neighbors=20)
    sc.tl.umap(adata)  # , svd_solver='arpack', n_comps=5)
    adata.obs["batch_type"] = [f"batch_{id}" for id in adata.obs["batch"]]
    sc.pl.umap(adata, color='batch_type')
    plt.legend(title='')
    plt.savefig(os.path.join(path, f'{title}_batch.png'))


def plot_adata(adata, plot_dir='', embed='', label='celltype', title='before-calibrationp'):
    plot_umap_batch(adata, path=plot_dir,title=title)
    plot_umap_celltype(adata, path=plot_dir,title=title)


def make_combinations_from_config(config):
    param_combinations = list(itertools.product(*config.values()))
    configurations = []
    for params in param_combinations:
        new_config = {key: value for key, value in zip(config.keys(), params)}
        configurations.append(new_config)

    return configurations


def sample_from_space(configurations, num_of_samples):
    list_configurations = random.sample(configurations, num_of_samples)

    for index, config in enumerate(list_configurations):
        expirement_name = f"expirement_{index}"
        config["expirement_name"] = expirement_name
        config["plots_dir"] = os.path.join(config["plots_dir"], expirement_name)
        config["save_weights"] = os.path.join(config["save_weights"], expirement_name)

    return list_configurations
