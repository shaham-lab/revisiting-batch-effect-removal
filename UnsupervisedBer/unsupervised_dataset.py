import torch
from torch.utils.data import Dataset


def organize_data(src_data, dest_data, src_labels=None, target_labels=None):
    """
    Organize source and destination data into a unified dataset format.
    
    This function combines source and destination data into a single dataset
    with batch indicators (0 for source, 1 for destination). It can handle
    both labeled and unlabeled data.
    
    Args:
        src_data (list or numpy.ndarray): Source batch data
        dest_data (list or numpy.ndarray): Destination batch data
        src_labels (list or numpy.ndarray, optional): Source batch labels
        target_labels (list or numpy.ndarray, optional): Target batch labels
    
    Returns:
        list: List of tuples (data, label, batch_id) where batch_id is 0 for source and 1 for destination
    """
    data_set = []

    if src_labels is None or target_labels == None:
        [data_set.append((src, None, torch.tensor(0, dtype=torch.float32))) for src in src_data]
        [data_set.append((dest, None, torch.tensor(1., dtype=torch.float32))) for dest in dest_data]
    else:
        [data_set.append((src, src_labels[index], torch.tensor(0, dtype=torch.float32))) for index, src in enumerate(src_data)]
        [data_set.append((dest, target_labels[index], torch.tensor(1., dtype=torch.float32))) for index, dest in enumerate(dest_data)]

    return data_set


class UnsupervisedDataset(Dataset):
    """
    PyTorch Dataset for unsupervised batch effect removal.
    
    This dataset combines source and destination batch data into a single
    dataset with batch indicators. It's designed for unsupervised learning
    where the goal is to align batches without using cell type labels.
    
    Attributes:
        src_data (list or numpy.ndarray): Source batch data
        dest_data (list or numpy.ndarray): Destination batch data
        data (list): Organized dataset with batch indicators
    """
    
    def __init__(self, src_data, dest_data, src_labels=None, target_labels=None):
        """
        Initialize the unsupervised dataset.
        
        Args:
            src_data (list or numpy.ndarray): Source batch data
            dest_data (list or numpy.ndarray): Destination batch data
            src_labels (list or numpy.ndarray, optional): Source batch labels
            target_labels (list or numpy.ndarray, optional): Target batch labels
        """
        self.src_data = src_data
        self.dest_data = dest_data
        self.data = organize_data(src_data, dest_data, src_labels, target_labels)

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int or torch.Tensor): Index of the sample to retrieve
        
        Returns:
            tuple: (data, label, batch_id) where batch_id indicates source (0) or destination (1)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_item = self.data[idx]

        return data_item
