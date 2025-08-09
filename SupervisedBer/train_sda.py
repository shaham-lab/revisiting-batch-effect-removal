import os
from statistics import mean
import scanpy as sc

import torch
from tqdm import tqdm
from focal_loss.focal_loss import FocalLoss

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from sda_utils import mmd
# from MMD import MMDLoss
from domain_adaption_net import Net
from expirements.load import assign_number_to_labels
from expirements.utils import plot_adata
from metric import silhouette_coeff_ASW
from plot_data import plot_umap_batch, plot_umap_celltype
# from plot_data import get_pca_data, plot_pca_data, plot_pca_data_cdca
# from pre_procesing.train_reduce_dim import pre_processing
from sda_utils import get_cdca_term, mmd, dsne_loss, ccsa_loss, calculate_f1_score, \
    calculate_auc_score, get_one_hot_encoding
import numpy as np
from sda_datasets import TargetDataset, CdcaDataset, ForeverDataIterator  # , SrcDataset, ForeverDataIterator

os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

# from sinkhorn import SinkhornSolver

from torch import nn
import random


# from sinkhorn import SinkhornSolver


def get_absulute_gradient(net):
    gradients = []
    for param in net.parameters():
        gradients.append(param.grad)

    gradient_norm = torch.sqrt(sum([torch.norm(grad) ** 2 for grad in gradients[:-2]]))
    return gradient_norm


def get_mutal_information(batch_y, mask0, mask1):
    unique_tensor1 = torch.unique(batch_y[mask0])
    unique_tensor2 = torch.unique(batch_y[mask1])

    # Convert tensors to sets
    set1 = set(unique_tensor1.tolist())
    set2 = set(unique_tensor2.tolist())

    # Perform set intersection

    intersection = set1 & set2
    # Convert the intersection set to a tensor for element-wise comparison
    insersection_cells = torch.tensor(list(intersection))
    # Use torch.isin to check if elements of 'tensor' are in 'intersection_tensor'
    mutal_information = torch.isin(batch_y, insersection_cells)
    return mutal_information


def compute_class_weights(labels):
    labels_list = labels.tolist()

    label_counts = Counter(labels_list)
    total_samples = len(labels_list)
    class_weights = {cls: total_samples / count for cls, count in label_counts.items()}
    return class_weights


def train_sda(cdca_dataloader, cdca_test, class_number
              , net, optimizer, criterion, lr_scheduler, train_size,
              loss_type, writer, coef,coef_uda, gpu_flag=False, epoch=20):
    cdca_dataloader_forever = ForeverDataIterator(cdca_dataloader)
    # target_dataloader_forever = ForeverDataIterator(target_dataloader)
    loss_type = loss_type
    # sinkhorn = SinkhornSolver()
    # mmd = MMDLoss()
    # net.to("cpu")
    for index in tqdm(range(epoch)):
        net.train()
        gradients_epoch = []
        counter = 0
        total_loss = 0
        epoch_loss = []
        while counter < len(cdca_dataloader):
            data, labels, batch_id = next(cdca_dataloader_forever)
            # target_data, target_labels = next(target_dataloader_forever)
            # print(compute_class_weights(labels))
            mask_0 = batch_id == 0
            mask_1 = batch_id == 1

            src_data = data[mask_0].detach()
            # src_data += torch.normal(mean=0, std=0.1, size=(src_data.shape[0], src_data.shape[1]))

            target_data = data[mask_1].detach()
            # target_data += torch.normal(mean=0, std=0.1, size=(target_data.shape[0], target_data.shape[1]))

            src_labels = labels[mask_0].detach()
            target_labels = labels[mask_1].detach()

            class_number = class_number  # max(len(np.unique(src_labels)), len(np.unique(target_labels)))
            src_data, target_data = src_data.clone().detach(), target_data.clone().detach()

            data_labels_encoding = torch.nn.functional.one_hot(labels.to(torch.int64), class_number)

            src_labels_encoding = torch.nn.functional.one_hot(src_labels.to(torch.int64), class_number)
            target_labels_encoding = torch.nn.functional.one_hot(target_labels.to(torch.int64), class_number)
            # src_labels = torch.clamp(src_labels, min=1e-3, max=1 - 1e-3).type(torch.float32).detach()
            # target_labels = torch.clamp(target_labels, min=1e-3, max=1 - 1e-3).type(torch.float32).detach()
            data_pred, data_featue = net(data.float())
            src_pred, src_feature = data_pred[mask_0], data_featue[mask_0]
            tgt_pred, tgt_feature = data_pred[mask_1], data_featue[mask_1]
            src_pred = src_pred.type(torch.float32)
            tgt_pred = tgt_pred.type(torch.float32)

            # -----Explicit losses-----
            u = mask_0.sum() / mask_0.shape[0]
            # loss_s = criterion(src_pred, src_labels_encoding.type(torch.float32))
            # loss_t = criterion(tgt_pred, target_labels_encoding.type(torch.float32))
            loss_s = criterion(src_pred, src_labels.type(torch.float32))
            loss_t = criterion(tgt_pred, target_labels.type(torch.float32))
            # loss_uda = mmd(src_feature, tgt_feature)
            if loss_type == "source":
                loss = loss_s / loss_s.item()
                loss_value = loss_s.item()
            if loss_type == "target":
                loss = loss_t / loss_t.item()
                loss_value = loss_t.item()

            if loss_type == "s&t":
                loss = loss_t / loss_t.item() + loss_s / loss_s.item()
                loss_value = loss_t.item() + loss_s.item()

            if loss_type == "s&t&u":
                loss_uda = mmd(src_feature, tgt_feature)

                loss = loss_t / loss_t.item() + loss_s / loss_s.item() + loss_uda / loss_uda.item()
                loss_value = loss_t.item() + loss_s.item() + loss_uda.item()

            if loss_type == "s&t&u&c":
                mutal_cells_mask = get_mutal_information(labels, mask_0, mask_1)
                mutal_cells_mask_0 = torch.logical_and(mutal_cells_mask, mask_0)
                mutal_cells_mask_1 = torch.logical_and(mutal_cells_mask, mask_1)

                src_mutal_feature = data_featue[mutal_cells_mask_0]
                target_mutal_feature = data_featue[mutal_cells_mask_1]
                src_mutal_labels = data_labels_encoding[mutal_cells_mask_0]
                target_mutal_labels = data_labels_encoding[mutal_cells_mask_1]
                loss_uda = mmd(src_mutal_feature, target_mutal_feature)
                labels_t_s, labels_s_t, labels_s_s, labels_t_t = get_cdca_term(src_mutal_feature, target_mutal_feature,
                                                                               src_mutal_labels,
                                                                               target_mutal_labels,
                                                                               n_classes=class_number)
                loss_t0_cdca = nn.L1Loss()(labels_s_t.type(torch.float32).squeeze(),
                                           src_mutal_labels.type(torch.float32))
                loss_t1_cdca = nn.L1Loss()(labels_t_s.type(torch.float32).squeeze(),
                                           target_mutal_labels.type(torch.float32))
                loss_cdca = loss_t0_cdca + loss_t1_cdca
                loss = loss_t + loss_s + coef * loss_cdca + coef_uda*loss_uda  # + 10*loss_cdca  # loss_t / loss_t.item() + loss_s / loss_s.item() + 0 * loss_uda / loss_uda.item() + 0 * loss_cdca / loss_cdca.item()
                loss_value = loss_t.item() + loss_s.item() + loss_uda.item() + loss_cdca.item()  # + 10 * loss_cdca.item()  #
                # +loss_uda.item()# + loss_uda.item()

            if loss_type == 'dSNE':
                loss_dsne = dsne_loss(src_feature, src_labels, tgt_feature, target_labels)
                loss = loss_s / loss_s.item() + loss_t / loss_t.item() + loss_dsne / loss_dsne.item()
                loss_value = loss_s.item() + loss_t.item() + loss_dsne.item()

            if loss_type == 'CCSA':
                loss_csca = ccsa_loss(src_feature, tgt_feature,
                                      (src_labels == target_labels).float())
                loss = loss_s / loss_s.item() + loss_csca / loss_csca.item()
                loss_value = loss_s.item() + loss_csca.item()

            counter += 1
            epoch_loss.append(loss_value)

            optimizer.zero_grad()
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
            gradients_epoch.append(get_absulute_gradient(net))

        total_loss = mean(epoch_loss)
        f1_score = calculate_f1_score(cdca_test, net)

        writer.add_scalar(f'F1 Score class', f1_score, global_step=index)

        # print(f"gradient: {max(gradients_epoch)}")
        # print(total_loss)
        writer.add_scalar(f"Loss/{loss_type}/train-{train_size}", total_loss, global_step=index)
        writer.add_scalar(f"Graidents/{loss_type}/train-{train_size}", max(gradients_epoch), global_step=index)
        lr_scheduler.step()
    # net.to('cpu')
    return net


from collections import Counter


def compute_class_weights(labels):
    labels_list = labels

    # Count the occurrences of each class
    label_counts = Counter(labels_list)

    # Get total number of samples
    total_samples = len(labels_list)

    # Calculate the weight for each class
    class_weights = {cls: total_samples / count for cls, count in label_counts.items()}

    # Convert to tensor for use in PyTorch
    class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32)

    return class_weights_tensor


def get_count_class(images, nclasses):
    count_per_class = [0] * nclasses
    for _, image_class, _ in images:
        count_per_class[image_class] += 1
    return count_per_class


def make_weights_for_balanced_classes(images, nclasses):
    n_images = len(images)
    count_per_class = get_count_class(images, nclasses)
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        if count_per_class[i] != 0:
            weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, (image, image_class, batch_id) in enumerate(images):
        weights[idx] = weight_per_class[image_class]
    return weights


def cdca_alignment(config, adata1, adata2, number_to_label, embed='', resample_data="resample"):
    writer = SummaryWriter(log_dir=f'runs/{config["experiment_name"]}')
    test_size = config["test_size"]
    if embed != '':
        src_data_without_labels, target_data_without_labels = adata1.obsm[embed], adata2.obsm[embed]
    else:
        src_data_without_labels, target_data_without_labels = adata1.X, adata2.X

    labels_src = np.array(adata1.obs["encoding-celltype"])
    labels_target = np.array(adata2.obs["encoding-celltype"])

    batch_id = np.concatenate((np.array([0] * len(labels_src)), np.array([1] * len(labels_target))))
    labels = np.concatenate((labels_src, labels_target))
    cells = np.concatenate((src_data_without_labels, target_data_without_labels), axis=0)
    print(cells.shape)
    print(f"number of celltype src:{np.unique(labels_src)}")
    print(f"number of celltype target:{np.unique(labels_target)}")

    class_number = len(np.unique(labels))
    cdca_dataset = CdcaDataset(cells, labels, batch_id, class_number)  # , class_number )# , method=resample_data)
    # target_dataset = TargetDataset(target_data_without_labels, labels_target)

    generator = torch.Generator()
    generator.manual_seed(0)
    cdca_test, cdca_train = torch.utils.data.random_split(cdca_dataset, [int(len(
        cdca_dataset) * test_size),
                                                                         len(cdca_dataset) - int(
                                                                             len(cdca_dataset) * test_size)]
                                                          , generator=generator)
    # weights_cdca = make_weights_for_balanced_classes(cdca_train, class_number)
    # weights_cdca = torch.DoubleTensor(weights_cdca)
    # sampler_cdca = torch.utils.data.sampler.WeightedRandomSampler(weights_cdca, len(weights_cdca))
    #
    # # train_set_target_dataset = TargetDataset(train_data, train_labels)  # , class_number)
    # , sampler=sampler_cdca
    cdca_dataloader = DataLoader(dataset=cdca_train, batch_size=int(config["batch_size"]),
                                 drop_last=True)
    weights = compute_class_weights(labels)
    print("Class Weights:", weights)

    # Step 3: Create a CrossEntropyLoss with weights
    criterion = nn.CrossEntropyLoss(weight=weights)

    criterion =  FocalLoss(gamma=0.7, weights=weights)

    config["input_dim"] = adata1.shape[1]

    net = Net(config["input_dim"],
              config["hidden_dim"],
              config["drop_prob"],
              config["hidden_layers"], class_num=class_number)

    optimizer = torch.optim.Adam(net.parameters()
                                 , lr=config["lr"], weight_decay=0.2)
    lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    net = train_sda(cdca_dataloader, cdca_test, class_number, net, optimizer, criterion,
                    lr_scheduler, train_size=1 - config["test_size"],
                    loss_type=config["loss_type"],
                    epoch=config["epochs"], coef=config["coef"],coef_uda=config["coef_uda"], writer=writer)

    _, embed_cells = net(torch.tensor(cells).float())
    adata_cdca = sc.AnnData(X=embed_cells.detach().numpy())
    adata_cdca.obsm['X_emb'] = embed_cells.detach().numpy()
    adata_cdca.obs['batch'] = batch_id
    adata_cdca.obs['batch'] = adata_cdca.obs['batch'].astype("category")
    # adata.obs['encoding-celltype'] = labels
    adata_cdca.obs['celltype'] = labels # assign_number_to_labels(labels, number_to_label)
    adata_cdca.obs['celltype'] = adata_cdca.obs['celltype'].astype("category")
    plot_adata(adata_cdca, plot_dir=config["plots_dir"],
               title='after-calibration-ours')

    # silhouette_coeff_ASW(adata, embed='').to_csv(os.path.join(config["plots_dir"],
    #                                                           "ASW_adata_ours.csv"))
    sc.tl.pca(adata_cdca, svd_solver='arpack', n_comps=20)
    adata_cdca.obsm['X_pca'] *= -1  # multiply by -1 to match Seurat

    # adata = adata.transpose()
    # silhouette_coeff_ASW(adata, method_use='raw', embed='')
    writer.flush()

    writer.close()

    return adata_cdca
