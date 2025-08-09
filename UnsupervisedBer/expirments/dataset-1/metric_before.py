from dataset_reader.read_dataset_1 import get_dataset
from metrics import calculate_ari, calculate_ASW, calulate_ari_nmi

# from scDML.scDML.metrics import calulate_ari_nmi

if __name__ == "__main__":
    adata = get_dataset()
    calculate_ari(
       adata=adata,
       cpcs=list(range(20)),  # First 10 PCs
       method_use='example_method',
       celltypelb='celltype',
       batchlb='batch'
    )
    print(calulate_ari_nmi(adata))
    # calculate_ASW(adata, method_use='raw', save_dir='', save_fn='', percent_extract=0.8).to_csv("ASW_adata_src_calibrated_target.csv")
