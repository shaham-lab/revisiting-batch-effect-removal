from pathlib import Path

from expirments.load import load_to_adata_shaham_dataset
from expirments.utils import plot_adata

src_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person2Day1_baseline.csv'
target_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person2Day2_baseline.csv'
src_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person2Day1_baseline_label.csv'
target_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person2Day2_baseline_label.csv'


def get_dataset():
    adata = load_to_adata_shaham_dataset(src_path, target_path, src_path_label, target_path_label)

    return adata


if __name__ == "__main__":
    filepath = r"adata_p2_baseline_norm.h5ad"
    filepath = Path(filepath)

    adata = get_dataset()
    # plot_adata(adata, plot_dir='', title='before-calibrationp')

    print(f'Write anndata to {filepath}')
    adata.write(filepath)
