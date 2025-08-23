import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import f1_score, classification_report


def get_scale_fac(d, discard_diag=True):
    """
    Calculate scale factor for kernel functions based on pairwise distances.
    
    This function computes a scale factor for Gaussian kernels by finding the
    median of the k-nearest neighbor distances. The scale factor helps determine
    the bandwidth of the kernel.
    
    Args:
        d (torch.Tensor): Pairwise distance matrix
        discard_diag (bool): Whether to discard diagonal elements (True for cross-batch distances)
    
    Returns:
        torch.Tensor: Scale factor for kernel computation
    """
    # when d=pdist(x,y), set discard_diag to True, when d=pdist(x,x), set discard_diag to False.
    value, index = torch.sort(d, dim=1)
    nn = value[:, :20]
    scale = torch.median(torch.median(nn, dim=1)[0])
    # if discard_diag:
    #     [min_per_row, _] = torch.median(d + torch.max(d) * torch.eye(n=d.shape[0], m=d.shape[1]).to(device), dim=1)
    # else:
    #     [min_per_row, _] = torch.median(d, dim=1)
    return scale


def get_weight_matrix(src_features,target_features):
    """
    Compute attention weight matrix between source and target features.
    
    This function calculates pairwise distances between source and target features,
    applies a kernel function, and computes softmax weights for attention mechanisms.
    
    Args:
        src_features (torch.Tensor): Source batch features
        target_features (torch.Tensor): Target batch features
    
    Returns:
        torch.Tensor: Weight matrix for cross-batch attention
    """
    pairwise_distances = torch.cdist(src_features,target_features, p=2.0)
    kernel_matrix = t_kernel(pairwise_distances)
    k_kernel_matrix = kernel_matrix  # * weight_matrix
    weight_matrix = softmax_torch(k_kernel_matrix)

    return weight_matrix


def get_one_hot_encoding(labels, n_classes):
    """
    Convert integer labels to one-hot encoded format.
    
    Args:
        labels (torch.Tensor): Integer labels
        n_classes (int): Number of classes
    
    Returns:
        torch.Tensor: One-hot encoded labels
    """
    one_hots_label = torch.zeros(len(labels), n_classes).to(labels.device)
    one_hots_label[range(len(labels)), labels.type(torch.int)] = 1

    return one_hots_label


## CCSA
def ccsa_loss(x, y, class_eq):
    """
    Compute Contrastive Semantic Alignment (CCSA) loss.
    
    CCSA encourages features from the same class to be close and features from
    different classes to be separated by a margin, regardless of their batch origin.
    
    Args:
        x (torch.Tensor): Source batch features
        y (torch.Tensor): Target batch features
        class_eq (torch.Tensor): Binary tensor indicating if pairs belong to same class
    
    Returns:
        torch.Tensor: CCSA loss value
    """
    margin = 1
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    y = torch.nn.functional.normalize(y, p=2, dim=1)
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)  # if eqaul classes: penalize dist
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)  # if different - penalize if larger than margin
    return loss.mean()


## dSNE
def dsne_loss(src_feature, src_labels, tgt_feature, target_labels):
    """
    Compute Deep Siamese Network Embedding (dSNE) loss.
    
    dSNE loss ensures that the maximum intra-class distance is smaller than the
    minimum inter-class distance by a margin, promoting better class separation
    in the learned embedding space.
    
    Args:
        src_feature (torch.Tensor): Source batch features
        src_labels (torch.Tensor): Source batch labels
        tgt_feature (torch.Tensor): Target batch features
        target_labels (torch.Tensor): Target batch labels
    
    Returns:
        torch.Tensor: dSNE loss value
    """
    bs_tgt, bs_src = len(src_feature), len(tgt_feature)
    embed_size = src_feature.shape[1]
    margin = 1
    src_feature = torch.nn.functional.normalize(src_feature, p=2, dim=1)
    tgt_feature = torch.nn.functional.normalize(tgt_feature, p=2, dim=1)

    fts_rpt = torch.broadcast_to(src_feature.unsqueeze(dim=0), size=(bs_tgt, bs_src, embed_size))
    ftt_rpt = torch.broadcast_to(tgt_feature.unsqueeze(dim=1), size=(bs_tgt, bs_src, embed_size))
    dists = torch.sum(torch.square(ftt_rpt - fts_rpt), axis=2)
    yt_rpt = torch.broadcast_to(target_labels.unsqueeze(dim=1), size=(bs_tgt, bs_src)).type(torch.int32)
    ys_rpt = torch.broadcast_to(src_labels.unsqueeze(dim=0), size=(bs_tgt, bs_src)).type(torch.int32)

    y_same = ((yt_rpt - ys_rpt) == 0)
    y_diff = torch.logical_not(y_same)

    intra_cls_dists = dists * y_same
    inter_cls_dists = dists * y_diff
    (max_dists, _) = torch.max(dists, axis=1, keepdims=True)
    max_dists = max_dists.expand((bs_tgt, bs_src))
    revised_inter_cls_dists = torch.where(y_same, max_dists, inter_cls_dists)
    (max_intra_cls_dist, _) = torch.max(intra_cls_dists, axis=1)
    (min_inter_cls_dist, _) = torch.min(revised_inter_cls_dists, axis=1)
    # loss = torch.sum(torch.relu(max_intra_cls_dist - min_inter_cls_dist + margin))
    loss = torch.mean(torch.relu(max_intra_cls_dist - min_inter_cls_dist + margin))
    return loss


# cdca loss


def get_one_hot_encoding(labels, n_classes):
    """
    Convert integer labels to one-hot encoded format (numpy version).
    
    Args:
        labels (numpy.ndarray or scalar): Integer labels
        n_classes (int): Number of classes
    
    Returns:
        numpy.ndarray: One-hot encoded labels
    """
    if not isinstance(labels, np.ndarray):
        labels = np.array([labels])
    # else torch.is_tensor(labels):
    one_hot_labels = np.zeros((len(labels), n_classes))
    one_hot_labels[range(len(labels)), labels] = 1

    return one_hot_labels


def t_kernel(values):
    """
    Apply t-kernel transformation to distance values.
    
    The t-kernel is a simple kernel function that transforms distances
    using the formula 1/(1 + values), creating a similarity measure.
    
    Args:
        values (torch.Tensor): Distance values
    
    Returns:
        torch.Tensor: Kernel-transformed values
    """
    kernel_t = 1 / (1 + values)

    return kernel_t


def gaussian_kernel(values, scale):
    """
    Apply Gaussian kernel transformation to distance values.
    
    The Gaussian kernel transforms distances using the formula
    exp(-values / scale), creating a similarity measure with bandwidth
    controlled by the scale parameter.
    
    Args:
        values (torch.Tensor): Distance values
        scale (torch.Tensor): Scale parameter for the Gaussian kernel
    
    Returns:
        torch.Tensor: Gaussian kernel-transformed values
    """
    scale_fac_i = scale
    kernel_gaussian = torch.exp(-values / (scale_fac_i.unsqueeze(1)).detach())

    return kernel_gaussian


def get_scale_fac(d):
    """
    Calculate scale factor for Gaussian kernel based on pairwise distances.
    
    This function computes a scale factor for Gaussian kernels by finding the
    median of the pairwise distances. The scale factor helps determine the
    bandwidth of the kernel.
    
    Args:
        d (torch.Tensor): Pairwise distance matrix
    
    Returns:
        torch.Tensor: Scale factor for kernel computation
    """
    # when d=pdist(x,y), set discard_diag to True, when d=pdist(x,x), set discard_diag to False.
    [median_row, _] = torch.median(d + torch.max(d) * torch.eye(n=d.shape[0], m=d.shape[1]), dim=1)

    return median_row


def get_weight_matrix_attn(src_feature, tgt_feature):
    """
    Compute attention weight matrix between source and target features using softmax.
    
    This function normalizes source and target features and computes a softmax
    weight matrix based on their dot product.
    
    Args:
        src_feature (torch.Tensor): Source batch features
        tgt_feature (torch.Tensor): Target batch features
    
    Returns:
        torch.Tensor: Weight matrix for cross-batch attention
    """
    src_feature = torch.nn.functional.normalize(src_feature, p=2, dim=1)
    tgt_feature = torch.nn.functional.normalize(tgt_feature, p=2, dim=1)

    weight_matrix = torch.nn.functional.softmax(src_feature @ tgt_feature.t() , dim=-1)

    return weight_matrix


def get_weight_matrix(src_label, tgt_label):
    """
    Compute attention weight matrix between source and target labels using t-kernel.
    
    This function calculates pairwise distances between source and target labels,
    applies a t-kernel transformation, and computes a weight matrix using
    diagonal normalization.
    
    Args:
        src_label (torch.Tensor): Source batch labels
        tgt_label (torch.Tensor): Target batch labels
    
    Returns:
        torch.Tensor: Weight matrix for cross-batch attention
    """
    pairwise_distances = torch.cdist(src_label, tgt_label, p=2.0)
    kernel_t = t_kernel(pairwise_distances)
    k_kernel_matrix = kernel_t
    weight_matrix = torch.diag(1 / (torch.sum(k_kernel_matrix, dim=1) + 1e-5)) @ k_kernel_matrix

    return weight_matrix


def get_gaussian_weight_matrix(src_label, tgt_label):
    """
    Compute attention weight matrix between source and target labels using Gaussian kernel.
    
    This function calculates pairwise distances between source and target labels,
    applies a Gaussian kernel transformation, and computes a weight matrix using
    diagonal normalization.
    
    Args:
        src_label (torch.Tensor): Source batch labels
        tgt_label (torch.Tensor): Target batch labels
    
    Returns:
        torch.Tensor: Weight matrix for cross-batch attention
    """
    pairwise_distances = torch.cdist(src_label, tgt_label, p=2.0)
    scale = get_scale_fac(pairwise_distances)
    kernel_gaussian = gaussian_kernel(pairwise_distances,scale)
    k_kernel_matrix = kernel_gaussian
    weight_matrix = torch.diag(1 / (torch.sum(k_kernel_matrix, dim=1) + 1e-4)) @ k_kernel_matrix

    return weight_matrix

def get_cdca_term(src_feature, tgt_feature, src_label, tgt_label,
                  n_classes=2, nearest_neighbours=50):
    """
    Compute terms for Cross-Domain Cross-Attention (CDCA) loss.
    
    This function computes attention matrices for different pairs of source and
    target features and labels, and returns the attention terms for CDCA loss.
    
    Args:
        src_feature (torch.Tensor): Source batch features
        tgt_feature (torch.Tensor): Target batch features
        src_label (torch.Tensor): Source batch labels
        tgt_label (torch.Tensor): Target batch labels
        n_classes (int): Number of classes
        nearest_neighbours (int): Number of nearest neighbors for scale calculation
    
    Returns:
        tuple: Attention terms for CDCA loss
    """
    attention_s_t = get_weight_matrix(src_feature, tgt_feature)
    attention_t_s = get_weight_matrix(tgt_feature, src_feature)
    attention_s_s = get_weight_matrix(src_feature, src_feature)
    attention_t_t = get_weight_matrix(tgt_feature, tgt_feature)
    src_label = src_label.type(torch.float32)
    tgt_label = tgt_label.type(torch.float32)
    labels_t_s = attention_t_s @ src_label
    labels_s_t = attention_s_t @ tgt_label
    labels_s_s = attention_s_s @ src_label
    labels_t_t = attention_t_t @ tgt_label

    return labels_t_s, labels_s_t, labels_s_s, labels_t_t


def mmd(x, y, mmd_kernel_bandwidth=[0.1, 0.5, 1, 2]):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """

    nx = x.shape[0]
    ny = y.shape[0]
    dxx = torch.cdist(x, x) ** 2  # (x_i-x_j)^2
    dyy = torch.cdist(y, y) ** 2  # (y_i-y_j)^2
    dxy = torch.cdist(x, y) ** 2  # (x_i-y_j)^2
    device = x.device if x.is_cuda else torch.device("cpu")
    XX, YY, XY = (torch.zeros(dxx.shape).to(device),
                  torch.zeros(dyy.shape).to(device),
                  torch.zeros(dxy.shape).to(device))

    for a in mmd_kernel_bandwidth:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)

    return torch.sum(XX) / (nx ** 2) + torch.sum(YY) / (ny ** 2) - 2. * torch.sum(XY) / (nx * ny)


def normalize_columns(tensor):
    """Normalizes all the columns in a 2D tensor.

    Args:
        tensor: The 2D tensor to normalize.

      Returns:
        The normalized 2D tensor.
      """

    # Get the maximum value of each column.
    max_values = tensor.max(dim=0)

    # Divide each column by the maximum value.
    normalized_tensor = tensor / max_values

    return normalized_tensor


def average(lst):
    return sum(lst) / len(lst)


def weighted_bce_loss(predictions, targets):
    neg_weight = targets.sum() / targets.numel() + 1e-3
    pos_weight = 1 - neg_weight + 1e-3

    loss = pos_weight * targets * torch.log(predictions) + neg_weight * (1 - targets) * torch.log(
        1 - predictions)
    loss = torch.neg(torch.sum(loss))
    return loss


def gaussian_kernel(values, scale):
    """
    Apply Gaussian kernel transformation to distance values (numpy version).
    
    The Gaussian kernel transforms distances using the formula
    exp(-values / scale), creating a similarity measure with bandwidth
    controlled by the scale parameter.
    
    Args:
        values (numpy.ndarray or scalar): Distance values
        scale (numpy.ndarray or scalar): Scale parameter for the Gaussian kernel
    
    Returns:
        numpy.ndarray: Gaussian kernel-transformed values
    """
    scale_fac_i = scale
    kernel_gaussian = torch.exp(-values / (scale_fac_i.unsqueeze(1)))

    return kernel_gaussian


def t_kernel(values):
    """
    Apply t-kernel transformation to distance values (numpy version).
    
    The t-kernel is a simple kernel function that transforms distances
    using the formula 1/(1 + values), creating a similarity measure.
    
    Args:
        values (numpy.ndarray or scalar): Distance values
    
    Returns:
        numpy.ndarray: Kernel-transformed values
    """
    kernel_t = 1 / (1 + values)

    return kernel_t


def make_onehot_to_value(x):
    return torch.argmax(x, dim=1)


def calculate_f1_score(dataset_test, net):
    """
    Calculate F1 score for a given dataset and model.
    
    This function evaluates the F1 score of a model on a test dataset,
    including source and target domains.
    
    Args:
        dataset_test (list): A list containing images, labels, and batch IDs.
        net (torch.nn.Module): The model to evaluate.
    
    Returns:
        float: Mean F1 score across all classes.
    """
    net.eval()
    images = torch.tensor(dataset_test[:][0])
    labels = torch.tensor(dataset_test[:][1])
    batch_id = torch.tensor(dataset_test[:][2])
    mask_target = batch_id == 1
    images = images.type(torch.float32)[mask_target]
    # images += torch.normal(mean=0, std=0.1, size=(images.shape[0], images.shape[1]))

    y_labels = labels.type(torch.float32).numpy()[mask_target]

    # Iterate over the dataloader.
    # Get the predictions.
    y_predictions, features = net(images)

    y_predictions = make_onehot_to_value(y_predictions).detach().numpy()
    # Get the true positives, false positives, and false negatives.
    f1 = f1_score(y_labels, y_predictions, average=None)
    # print(classification_report(y_labels, y_predictions, digits=4))

    return np.array(f1).mean()


def softmax_torch(x):  # Assuming x has atleast 2 dimensions
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True) + 1e-4
    probs = x_exp / x_exp_sum
    return probs


def softmax_sin(x):  # Assuming x has atleast 2 dimensions
    x_exp = torch.exp(torch.sin(x))
    x_sum = (torch.sum(x_exp, 1))
    M = x_sum.unsqueeze(dim=-1) - x_exp
    probs = x_exp / (M + x_exp)
    return probs


import torch
import numpy as np

if __name__ == "__main__":
    x = torch.randn(4, 3)
    std_pytorch_softmax = torch.nn.functional.softmax(x)
    pytorch_impl = softmax_sin(x)
    print(pytorch_impl)
    print(std_pytorch_softmax)


def calculate_auc_score(dataset_test, net):
    """
    Calculaes the F1 score of a model on a given dataloader.

    Args:
        dataloader: A dataloader for the test set.
        net: The model to evaluate.

    Returns:
        The F1 score of the model.
    """
    net.eval()
    images = dataset_test[:][0]
    labels = dataset_test[:][1]
    images = torch.tensor(images)
    labels = torch.tensor(labels)
    images = images.type(torch.float32)
    labels = labels.type(torch.float32)

    # Get the predictions.
    predictions, features = net(images)
    # predictions = torch.argmax(predictions,dim=1)
    auc = metrics.roc_auc_score(labels, predictions.detach().numpy(), multi_class='ovr')

    print(auc)
    return auc
