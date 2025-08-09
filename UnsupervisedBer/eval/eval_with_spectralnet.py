import numpy as np
import torch
from spectralnet import SpectralNet
from spectralnet import Metrics
import scanpy as sc
import glob
import os

from dataset_reader.read_dataset_2 import get_dataset


def eval_spectralnet(X, y):
    n_labels = len(set(y))
    X = torch.tensor(X).to(torch.float32)
    y = torch.tensor(y).to(torch.float32)
    spectralnet = SpectralNet(spectral_epochs=40, n_clusters=n_labels, spectral_hiddens=[1024, 1024, 512, n_labels])
    spectralnet.fit(torch.tensor(X), torch.tensor(y))  # X is the dataset and it should be a torch.Tensor
    cluster_assignments = spectralnet.predict(X)  # Get the final assignments to clusters

    nmi_score = Metrics.nmi_score(cluster_assignments, y)
    acc_score = Metrics.acc_score(cluster_assignments, y.numpy(), n_clusters=n_labels)
    print(f"ACC: {np.round(acc_score, 3)}")
    print(f"NMI: {np.round(nmi_score, 3)}")


if __name__ == "__main__":
    adata_orignal_space = get_dataset()
    path_code_dir = r"C:\Users\avrah\PycharmProjects\UnsuperVisedBer\expirments\plots\ours\dataset2-benchmark\code_space"
    extension = '**/*.h5ad'

    # Use glob to find all files with the specified extension
    files = glob.glob(os.path.join(path_code_dir, extension), recursive=True)
    print(files)
    # Print the list of files
    for file in files:
        adata = sc.read_h5ad(file)
        X = adata.X
        print(X.shape)
        if X.shape[1] == 2056:
            lables = adata.obs['celltype']
            eval_spectralnet(X, lables)
# ACC: 0.316 code space
# NMI: 0.233
