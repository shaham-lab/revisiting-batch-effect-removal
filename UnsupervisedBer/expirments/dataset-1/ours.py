import json
import os
from pathlib import Path

import numpy as np
import scib

from dataset_reader.read_dataset_1 import get_dataset
from expirments.load import assign_labels_to_numbers, make_adata_from_batches, get_batch_from_adata
from expirments.utils import sample_from_space, make_combinations_from_config, plot_adata
from main import ber_for_notebook


parent_dir = Path(r'C:/Users/avrah/PycharmProjects/UnsuperVisedBer/expirments')

config = {
    "hidden_dim_encoder": [1000,3000,2000],
    "hidden_dim_decoder": [1000,3000,2000],
    "code_dim": [128,256],

    "lr": [0.001,0.0005,0.0001],  # np.random.uniform(1e-2, 1e-1, size=5).tolist(),  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2,],  # or tune.choice([<list values>])
    "weight_decay": [ 0.15,0.2],  # or tune.choice([<list values>])
    "batch_size": [128,64,32],  # or tune.choice([<list values>])
    "drop_prob": [0.2,0.25],  # or tune.choice([<list values>])
    "scale": [False,True],
    "hvg": [False,True],
    "epochs": [40],
    "coef_1": [100,1000,10],
    "save_weights": [os.path.join(parent_dir, "weights/ber/dataset1-benchmark-new31/")],
    "plots_dir": [os.path.join(parent_dir, "plots/ours/dataset1-benchmark-new31/")]
}

dim_reduce_weights_path = os.path.join(config["save_weights"][0], "dim_reduce")

configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=10)

if __name__ == "__main__":
    adata_orignal = get_dataset()
    encoding_labels,number_to_label = assign_labels_to_numbers(adata_orignal.obs['celltype'])
    adata_orignal.obs['celltype'] = np.array(encoding_labels)
    encoding_labels,number_to_label = assign_labels_to_numbers(adata_orignal.obs['batch'])
    adata_orignal.obs['batch'] = np.array(encoding_labels)
    adata_orignal.obs['batch'] = adata_orignal.obs['batch'].astype(str).astype('category')

    from baseline_methods.mean_varince import batch_effect_correction


    adata1 = adata_orignal[adata_orignal.obs['batch'] == '0', :].copy()
    adata2 = adata_orignal[adata_orignal.obs['batch'] == '1', :].copy()
    batch_a_to_b = batch_effect_correction(adata1.X, adata2.X)
    adata_orignal = make_adata_from_batches(batch_a_to_b, adata2.X, adata1.obs['celltype'],
                                                          adata2.obs['celltype'])
    print(set(adata_orignal.obs['celltype']))
    print(set(adata_orignal.obs['batch']))
    adata_orignal.obs['batch'] = adata_orignal.obs['batch'].astype("category")

    # encoding_labels,number_to_label = assign_labels_to_numbers(adata_orignal.obs['celltype'])
    # adata_orignal.obs['celltype'] = np.array(encoding_labels)

    # Set the batch key for each cell

    for config in configurations:
        try:
            adata = adata_orignal.copy()
            if config["hvg"]==True:
                adata = scib.preprocessing.hvg_batch(
                    adata,
                    batch_key="batch",
                    target_genes=2000,
                    adataOut=True
                )
                adata.obs['batch'] = adata_orignal.obs['batch'].astype('category')

            if config['scale']==True:
                print("perform scale")
                adata = scib.preprocessing.scale_batch(adata, "batch")

                adata.obs['batch'] = adata_orignal.obs['batch'].astype('category')


            adata1,adata2 = get_batch_from_adata(adata)

            print(set(adata1.obs['celltype']))
            print(set(adata1.obs['batch']))

            os.makedirs(config["save_weights"], exist_ok=True)
            os.makedirs(config["plots_dir"], exist_ok=True)
            plot_adata(adata, plot_dir=config["plots_dir"], embed='X_pca', label='celltype', title='before-calibrationp')

            with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
                file.write(json.dumps(config))

            adata_code,adata_target_calibrated_src, adata_src_calibrated_target = ber_for_notebook(config, adata1=adata1,
                                                                                        adata2=adata2,return_in="original_space_and_code")
            plot_adata(adata_target_calibrated_src, plot_dir=config["plots_dir"],
                       title='after-calibration-target_calibrated_src')
            plot_adata(adata_src_calibrated_target, plot_dir=config["plots_dir"],
                       title='after-calibration-src_calibrated_target')
            plot_adata(adata_code, plot_dir=config["plots_dir"],
                       title='cdoe_space')

            adata_target_calibrated_src.write(os.path.join(config["plots_dir"],'after_calib_src_target.h5ad'))
            adata_src_calibrated_target.write(os.path.join(config["plots_dir"] , 'after_calib_target_src.h5ad'))
            adata_code.write(os.path.join(config["plots_dir"] , 'code_1.h5ad'))
        except Exception as e:
            pass