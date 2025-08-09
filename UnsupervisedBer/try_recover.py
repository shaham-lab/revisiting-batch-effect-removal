import scib
import torch
from sklearn.preprocessing import MinMaxScaler

from dataset_reader.read_dataset_1 import get_dataset
from expirments.load import make_adata_from_batches
from expirments.utils import plot_adata
from get_component_config import initialize_components
from main import get_data_calibrated
from unsupervised.ber_network import Net
config = {"hidden_dim_encoder": 20, "hidden_dim_decoder": 100, "code_dim": 256, "lr": 0.01, "dropout": 0.2, "weight_decay": 0.35, "batch_size": 128, "drop_prob": 0.2, "scale": True, "hvg": False, "epochs": 40, "coef_1": 2, "save_weights": r"C:\\Users\\avrah\\PycharmProjects\\UnsuperVisedBer\\expirments\\weights/ber/dataset1-benchmark/expirement_7", "plots_dir": "C:\\Users\\avrah\\PycharmProjects\\UnsuperVisedBer\\expirments\\plots/ours/dataset1-benchmark/expirement_7", "expirement_name": "expirement_7"}
if __name__ == "__main__":
    adata = get_dataset()
    adata = scib.preprocessing.scale_batch(adata, "batch")

    x =torch.tensor(adata.X)
    config["input_dim"] = adata.X.shape[1]
    batch = torch.tensor(adata.obs['batch'].astype(int))
    net, ind_discriminator, ae_optim, ind_disc_optim = initialize_components(config)
    path = r"C:\Users\avrah\PycharmProjects\UnsuperVisedBer\expirments\weights\ber\dataset1-benchmark\expirement_7\net.pt"
    state_dict = torch.load(path)
    net.load_state_dict(state_dict)
    net.eval()
    min_max_scaler_src = MinMaxScaler((-0.7, 0.7))
    min_max_scaler_target = MinMaxScaler((-0.7, 0.7))
    adata1 = adata[adata.obs['batch'] == "0", :].copy()
    adata2 = adata[adata.obs['batch'] == "1", :].copy()
    src_data_without_labels = min_max_scaler_src.fit_transform(adata1.X)
    target_data_without_labels = min_max_scaler_target.fit_transform(adata2.X)
    src = torch.tensor(src_data_without_labels).float()
    target = torch.tensor(target_data_without_labels).float()
    print("here")
    code_src, code_target, recon_src, recon_target, calibrated_src, calibrated_target = get_data_calibrated(src, target,
                                                                                                            net.encoder,
                                                                                                            net.decoder)
    adata_target_calibrated_src = make_adata_from_batches(target.detach().numpy(),
                                                          calibrated_src.detach().numpy(), adata2.obs['celltype'],
                                                          adata1.obs['celltype'])

    adata_target_calibrated_target = make_adata_from_batches(src.detach().numpy(),
                                                          calibrated_target.detach().numpy(), adata1.obs['celltype'],
                                                          adata2.obs['celltype'])

    plot_adata(adata_target_calibrated_src, plot_dir='',
               title='adata_target_calibrated_src')
    plot_adata(adata_target_calibrated_target, plot_dir='',
               title='adata_target_calibrated_target')
