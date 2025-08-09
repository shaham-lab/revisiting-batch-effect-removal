import itertools
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from expirements.load import get_batch_from_adata
from plot_data import get_pca_data, plot_scatter, plot_umap_batch, plot_umap_celltype


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
        experiment_name = f"{config['experiment_name']}_{index}"
        config["experiment_name"] = experiment_name
        config["plots_dir"] = os.path.join(config["plots_dir"], experiment_name)
        config["save_weights"] = os.path.join(config["save_weights"], experiment_name)

    return list_configurations
