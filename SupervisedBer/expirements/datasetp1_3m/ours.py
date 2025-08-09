import json
import os
from pathlib import Path
import scanpy as sc
import numpy as np

from expirements.load import load_to_adata_shaham_dataset, get_batch_from_adata, assign_labels_to_numbers
from expirements.utils import sample_from_space, make_combinations_from_config, plot_adata
from train_sda import cdca_alignment

src_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_3month.csv'
target_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_3month.csv'
src_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_3month_label.csv'
target_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_3month_label.csv'
parent_dir = Path(r'C:/Users/avrah/PycharmProjects/cdca/expirements')

config = {
    "loss_type": ["s&t&u&c"],
    "experiment_name": ["p1-s&t&u&c"],
    "hidden_dim": [25],
    "drop_prob": [0.2],
    "hidden_layers": [4],
    "lr": [0.01,0.001, 0.0001],  # np.random.uniform(1e-2, 1e-1, size=5).tolist(),  # Nuber of covariates in the data
    "test_size": [0.3],
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2],  # or tune.choice([<list values>])
    "weight_decay": [0.15, 0.25, 0.2],  # or tune.choice([<list values>])
    "batch_size": [128,256],  # or tune.choice([<list values>])
    "epochs": [50],
    "coef":[1,10,100],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset-p1_3m/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset-p1_3m/")]
}
configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=5)

if __name__ == "__main__":

    adata = load_to_adata_shaham_dataset(src_path, target_path, src_path_label, target_path_label)
    encoding_labels,number_to_label = assign_labels_to_numbers(adata.obs['celltype'])
    adata.obs['encoding-celltype'] = np.array(encoding_labels)
    print(adata.X.shape)
    adata1, adata2 = get_batch_from_adata(adata)

    for config in configurations:
        os.makedirs(config["save_weights"], exist_ok=True)
        os.makedirs(config["plots_dir"], exist_ok=True)
        plot_adata(adata, plot_dir=config["plots_dir"],embed='X_pca',label='celltype', title='before-calibrationp')
        #
        # sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)

        with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
            file.write(json.dumps(config))

        adata_calibration = cdca_alignment(config, adata1=adata1, adata2=adata2,number_to_label=number_to_label)

        adata_calibration.write(os.path.join(config["plots_dir"],'cdca_latent_1.h5ad'))
