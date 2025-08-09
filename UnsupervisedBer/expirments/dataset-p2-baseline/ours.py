import os

import json
import os


from expirments.utils import make_combinations_from_config, sample_from_space, plot_adata
from main import ber_for_notebook
from expirments.load import load_to_adata_shaham_dataset, get_batch_from_adata


from expirments.load import load_to_adata_shaham_dataset
from expirments.utils import make_combinations_from_config, sample_from_space
from expirments.load import load_to_adata_shaham_dataset
print(f"here {os.curdir}")
plots_dir = r"../plots/ours/dataset-p2-baseline"

os.makedirs(plots_dir, exist_ok=True)
src_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person2Day1_baseline.csv'
target_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person2Day2_baseline.csv'
src_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person2Day1_baseline_label.csv'
target_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person2Day2_baseline_label.csv'

#
# src_path = '../../data/Cytof/Person1Day1_baseline.csv'
# target_path = '../../data/Cytof/Person1Day2_baseline.csv'
# src_path_label = '../../data/Cytof/Person1Day1_baseline_label.csv'
# target_path_label = '../../data/Cytof/Person1Day2_baseline_label.csv'

config = {
    "hidden_dim_encoder": [10,20,50],
    "hidden_dim_decoder": [10,20,50],
    "code_dim": [52,26],  #
    "drop_prob": [0.2],  # or tune.choice([<list values>])

    "lr": [0.01,0.001, 0.005],  # Nuber of covariates in the data
    # or just tune.grid_search([<list of lists>])
    "dropout": [0.2],  # or tune.choice([<list values>])
    "weight_decay": [0.2],  # or tune.choice([<list values>])
    "batch_size": [64, 128],  # or tune.choice([<list values>])
    "epochs": [40],
    "coef_1": [1, 10, 5],
    "save_weights": [r"/weights/ber/dataset-p2-baseline/"],
    "plots_dir": [r"../plots/ours/dataset-p2-baseline/"]}


configurations = make_combinations_from_config(config)
configurations = sample_from_space(configurations, num_of_samples=8)

if __name__ == "__main__":
    adata = load_to_adata_shaham_dataset(src_path, target_path, src_path_label, target_path_label)
    # scanpy.pp.normalize_total(adata)

    for config in configurations:
        os.makedirs(config["save_weights"], exist_ok=True)
        os.makedirs(config["plots_dir"], exist_ok=True)
        # evaluate_dataset(adata).to_csv(
        #     os.path.join(config["plots_dir"], "orignal_adata.csv"))  # Set the batch key for each cell
        #
        plot_adata(adata, plot_dir=config["plots_dir"], title='before-calibrationp')
        with open(os.path.join(config["plots_dir"], "expirement.txt"), 'w') as file:
            file.write(json.dumps(config))
        adata1, adata2 = get_batch_from_adata(adata)
        # sc.pp.scale(adata1)
        # sc.pp.scale(adata2)

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
