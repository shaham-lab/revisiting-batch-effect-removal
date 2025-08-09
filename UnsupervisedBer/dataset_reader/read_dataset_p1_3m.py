from pathlib import Path

from expirments.load import load_to_adata_shaham_dataset
from expirments.utils import plot_adata

src_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_3month.csv'
target_path = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_3month.csv'
src_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day1_3month_label.csv'
target_path_label = r'C:\Users\avrah\OneDrive\שולחן העבודה\BER\data\Person1Day2_3month_label.csv'

def get_dataset():
    adata = load_to_adata_shaham_dataset(src_path, target_path, src_path_label, target_path_label)

    return adata

if __name__ == "__main__":

    filepath = r"adata_p1_3m_norm.h5ad"
    filepath = Path(filepath)

    adata = get_dataset()
    plot_adata(adata, plot_dir='', title='before-calibrationp')

    print(f'Write anndata to {filepath}')
    adata.write(filepath)