import copy
import os
import random
from statistics import median, stdev, mean

import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from expirments.load import make_adata_from_batches
from expirments.utils import apply_umap, apply_pca
from get_component_config import initialize_components
from pre_procesing.utils import load_and_pre_process_data
from unsupervised_dataset import UnsupervisedDataset
from unsupervised.autoencoder import Encoder
from unsupervised.ind_discrimnator import IndDiscriminator
from unsupervised.ber_network import DecoderBer, Net
from unsupervised.utils import indep_loss, eval_mmd, gradient_penalty, lr_scheduler, get_cdca_term


def get_data_calibrated(src_data, target_data, encoder, decoder):
    """
    Generate calibrated data by swapping batch indicators during decoding.
    
    This function encodes source and target data, then decodes them using
    the opposite batch indicator to create calibrated versions that should
    be more similar across batches.
    
    Args:
        src_data (torch.Tensor): Source batch data
        target_data (torch.Tensor): Target batch data
        encoder (nn.Module): Encoder network
        decoder (nn.Module): Batch-conditioned decoder network
    
    Returns:
        tuple: (code_src, code_target, recon_src, recon_target, calibrated_src, calibrated_target)
            - code_src: Encoded source data
            - code_target: Encoded target data
            - recon_src: Reconstructed source data (same batch indicator)
            - recon_target: Reconstructed target data (same batch indicator)
            - calibrated_src: Calibrated source data (opposite batch indicator)
            - calibrated_target: Calibrated target data (opposite batch indicator)
    """
    encoder.eval()
    decoder.eval()
    y_src = torch.zeros(src_data.shape[0])
    y_target = torch.ones(target_data.shape[0])
    code_src = encoder(src_data)
    code_target = encoder(target_data)
    recon_src, _ = decoder(code_src, y_src)
    _, recon_target = decoder(code_target, y_target)
    _, calibrated_src = decoder(code_src, 1 - y_src)
    calibrated_target, _ = decoder(code_target, 1 - y_target)

    return code_src, code_target, recon_src, recon_target, calibrated_src, calibrated_target


def validate(src, target, encoder, decoder):
    """
    Validate batch alignment using MMD (Maximum Mean Discrepancy).
    
    This function computes the MMD between calibrated data and original data
    to measure how well the batches have been aligned. Lower MMD indicates
    better alignment.
    
    Args:
        src (torch.Tensor): Source batch data
        target (torch.Tensor): Target batch data
        encoder (nn.Module): Encoder network
        decoder (nn.Module): Batch-conditioned decoder network
    
    Returns:
        torch.Tensor: MMD value indicating batch alignment quality
    """
    encoder.eval()
    decoder.eval()
    code_src, code_target, recon_src, recon_target, calibrated_src, calibrated_target = get_data_calibrated(src,
                                                                                                            target,
                                                                                                            encoder,
                                                                                                            decoder)

    calibrated_target_umap = torch.tensor(apply_pca(calibrated_target.detach().numpy()))
    calibrated_src_umap = torch.tensor(apply_pca(calibrated_src.detach().numpy()))
    src_umap = torch.tensor(apply_pca(src.detach().numpy()))
    target_umap = torch.tensor(apply_pca(target.detach().numpy()))
    # print(calibrated_target_umap.shape)
    # print(src_umap.shape)
    # print(calibrated_src_umap.shape)
    # print(target_umap.shape)
    mmd = min(eval_mmd(calibrated_target_umap, src_umap)[0], eval_mmd(calibrated_src_umap, target_umap)[0])
    return mmd


def get_mutal_information(batch_y, mask0, mask1):
    """
    Find cells that exist in both batches based on their labels.
    
    This function identifies cells that have the same cell type labels
    across different batches, which is useful for cross-domain alignment.
    
    Args:
        batch_y (torch.Tensor): Cell type labels for all cells
        mask0 (torch.Tensor): Boolean mask for first batch
        mask1 (torch.Tensor): Boolean mask for second batch
    
    Returns:
        torch.Tensor: Boolean mask indicating cells present in both batches
    """
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


def train2(src, target, data_loader, net, ind_discriminator, ae_optim, ind_disc_optim,
           config, dataset
           ):
    """
    Train the unsupervised batch effect removal network.
    
    This function implements the main training loop for unsupervised batch effect removal.
    It alternates between training the independence discriminator and the autoencoder,
    using adaptive loss coefficients based on MMD values.
    
    Args:
        src (torch.Tensor): Source batch data
        target (torch.Tensor): Target batch data
        data_loader (DataLoader): DataLoader for training data
        net (nn.Module): Autoencoder network (encoder + decoder)
        ind_discriminator (nn.Module): Independence discriminator
        ae_optim (torch.optim.Optimizer): Optimizer for autoencoder
        ind_disc_optim (torch.optim.Optimizer): Optimizer for discriminator
        config (dict): Configuration dictionary
        dataset (Dataset): Training dataset
    
    Returns:
        tuple: (net, recon_losses, independence_losses)
            - net: Trained autoencoder network
            - recon_losses: List of reconstruction losses per epoch
            - independence_losses: List of independence losses per epoch
    """
    dataset_x = torch.tensor([item[0] for item in dataset]).float()
    dataset_y = torch.tensor([item[1] for item in dataset])
    dataset_id = torch.tensor([item[2] for item in dataset])
    mask_0_dataset = dataset_id == 0
    mask_1_dataset = dataset_id == 1
    mutal_dataset_information_mask = get_mutal_information(dataset_y, mask_0_dataset, mask_1_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    recon_criterion = nn.MSELoss()
    recon_losses = []
    independence_losses = []
    coef = [1, config['coef_1'], 1]
    mmd_list = []

    l1_list = []
    l2_list = []
    best_mmd = torch.tensor(10000)
    smoothed_disc_loss = 0

    for epoch in tqdm(range(config["epochs"])):

        net.encoder.train(True)
        net.decoder.train(True)
        ind_discriminator.train(True)
        average_discriminator_loss = 0
        average_ae_loss = 0
        counter = 0
        for step, (batch_x, batch_y, batch_id) in enumerate(data_loader):
            counter += 1
            batch_x = batch_x.float().to(device=device).clone().detach()
            batch_id = batch_id.float().to(device=device).clone().detach()
            mask0 = batch_id == 0
            mask1 = batch_id == 1
            if torch.all(mask1) or torch.all(mask0):
                continue
            ind_discriminator.zero_grad()
            code_real = net.encoder(batch_x)
            ### doing the indepenence only on mutal information
            mutal_information_mask = get_mutal_information(batch_y, mask0, mask1)
            logist = ind_discriminator(code_real.detach())

            independence = coef[0] * indep_loss(logist, batch_id,
                                                should_be_dependent=True)
            independence_loss_value = independence.item()
            independence.backward()
            ind_disc_optim.step()
            average_discriminator_loss += abs(independence_loss_value)
            ##############56############### train autoencoder ###############
            if epoch % 1 == 0:
                net.encoder.zero_grad()
                net.decoder.zero_grad()

                code_real = net.encoder(batch_x).float()

                recon_batch_a, _ = net.decoder(code_real[mask0], batch_id[mask0])
                _, recon_batch_b = net.decoder(code_real[mask1], batch_id[mask1])

                recon_loss_a = recon_criterion(recon_batch_a, batch_x[mask0])
                recon_loss_b = recon_criterion(recon_batch_b, batch_x[mask1])

                logist = ind_discriminator(code_real.detach())
                # independence = indep_loss(logist[mutal_information_mask], batch_id[mutal_information_mask],
                #                           should_be_dependent=False)
                # normalize_code_1 = torch.nn.functional.normalize(code_real[mask0], p=2, dim=1)
                # normalize_code_2 = torch.nn.functional.normalize(code_real[mask1], p=2, dim=1)
                #
                # pq = torch.abs(normalize_code_1 @ normalize_code_2.T)
                # entropy_pq = torch.sum(pq)
                # entropy_pq = torch.sum(-((pq/4 + 1)) * torch.log2((pq/4 +1)))
                independence = indep_loss(logist, batch_id,
                                          should_be_dependent=False)
                ae_loss = coef[1] * (recon_loss_a / recon_loss_a.item() + recon_loss_b / recon_loss_b.item()) + coef[
                    2] * independence / independence.item() #+  entropy_pq / entropy_pq.item()
                ae_loss_value = recon_loss_a.item() + recon_loss_b.item()
                l1_list.append(recon_loss_a + recon_loss_b)
                l2_list.append(independence)

                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                ae_loss.backward()
                ae_optim.step()

                average_ae_loss += ae_loss_value

        if epoch % 3 == 0 and epoch > 0:
            net.encoder.eval()
            mutal_cells_mask_0 = torch.logical_and(mutal_dataset_information_mask, mask_0_dataset)
            mutal_cells_mask_1 = torch.logical_and(mutal_dataset_information_mask, mask_1_dataset)

            mmd = validate(dataset_x[mutal_cells_mask_0], dataset_x[mutal_cells_mask_1], net.encoder, net.decoder)
            # mmd = validate(src, target, net.encoder, net.decoder)
            mmd_code = eval_mmd(net.encoder(dataset_x[mutal_cells_mask_0]), net.encoder(dataset_x[mutal_cells_mask_1]))[
                0]
            # mmd_code = eval_mmd(net.encoder(src), net.encoder(target))[0]

            net.encoder.train()
            mmd_list.append(mmd_code.detach().numpy())
            if mmd < best_mmd:
                best_mmd = mmd
                print(best_mmd)
                net.save(config["save_weights"])

            # Save the best model

        if len(mmd_list) > 2:
            coef[0] = 1
            coef[1] = config["coef_1"] / np.std(np.array(mmd_list)) / np.mean(np.array(mmd_list))
            coef[2] = np.std(np.array(mmd_list)) / np.mean(np.array(mmd_list))
            print(f'Re-- {coef[1]}')
            print(f'In-- {coef[2]}')
            # else:
            #     coef[0] = 1
            #     coef[1] = 30  # / np.std(np.array(mmd_list)) / np.mean(np.array(mmd_list))
            #     coef[2] = 1 / np.std(np.array(mmd_list)) / np.mean(np.array(mmd_list)) + random.randint(0, 30)

        smoothed_disc_loss = 0.95 * smoothed_disc_loss + 0.05 * independence.item()
        current_lr = ind_disc_optim.param_groups[0]['lr']

        for param_group in ind_disc_optim.param_groups:
            param_group['lr'] = current_lr * lr_scheduler(smoothed_disc_loss, 0.63)

        recon_losses.append(average_ae_loss / counter)
        independence_losses.append(average_discriminator_loss / counter)

        recon_losses.append(average_ae_loss / counter)
        independence_losses.append(average_discriminator_loss / counter)

    print(f"----------{best_mmd}----------")

    return net, recon_losses, independence_losses


def ber_for_notebook(config, adata1, adata2, embed='', load_pre_weights='', return_in='original_space'):
    """
    Main function for unsupervised batch effect removal.
    
    This function implements the complete pipeline for unsupervised batch effect removal.
    It preprocesses data, trains the autoencoder with independence discriminator,
    and returns calibrated data in the specified space (original, code, or both).
    
    Args:
        config (dict): Configuration dictionary containing hyperparameters
        adata1 (AnnData): Source batch data
        adata2 (AnnData): Target batch data
        embed (str): Embedding key in adata.obsm to use (empty for raw data)
        load_pre_weights (str): Path to pre-trained weights (empty to train from scratch)
        return_in (str): Return format ('original_space', 'code_space', or 'original_space_and_code')
    
    Returns:
        AnnData or tuple: Calibrated data in specified format
            - If return_in == 'code_space': Returns AnnData with latent representations
            - If return_in == 'original_space': Returns tuple of (src_calibrated, target_calibrated)
            - If return_in == 'original_space_and_code': Returns tuple of (code, src_calibrated, target_calibrated)
    """
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    # min_max_scaler_src = StandardScaler()
    # min_max_scaler_target = StandardScaler()
    min_max_scaler_src = MinMaxScaler((-0.7, 0.7))
    min_max_scaler_target = MinMaxScaler((-0.7, 0.7))
    if embed == '':
        src_data_without_labels = adata1.X
        target_data_without_labels = adata2.X
    else:
        src_data_without_labels = adata1.obsm[embed]
        target_data_without_labels = adata2.obsm[embed]

    source_labels = torch.tensor(adata1.obs['celltype'])
    target_labels = torch.tensor(adata2.obs['celltype'])

    src_data_without_labels = min_max_scaler_src.fit_transform(src_data_without_labels)
    target_data_without_labels = min_max_scaler_target.fit_transform(target_data_without_labels)

    dataset = UnsupervisedDataset(src_data_without_labels, target_data_without_labels, source_labels, target_labels)
    src = torch.tensor(src_data_without_labels).float()
    target = torch.tensor(target_data_without_labels).float()
    train_loader = DataLoader(dataset, shuffle=True, batch_size=config["batch_size"], drop_last=True)
    config['input_dim'] = src_data_without_labels.shape[1]
    net, ind_discriminator, ae_optim, ind_disc_optim = initialize_components(config)

    if load_pre_weights == '':
        net, recon_losses, independence_losses = train2(src, target, train_loader, net,
                                                        ind_discriminator,
                                                        ae_optim, ind_disc_optim,
                                                        config, dataset)

        net.from_pretrain(os.path.join(config["save_weights"]))
    else:
        net.from_pretrain(load_pre_weights)

    net.eval()
    code_src, code_target, recon_src, recon_target, calibrated_src, calibrated_target = get_data_calibrated(src, target,
                                                                                                            net.encoder,
                                                                                                            net.decoder)
    adata_code = make_adata_from_batches(code_src.detach().numpy(),
                                         code_target.detach().numpy(),
                                         source_labels,
                                         target_labels
                                         )
    adata_code.obsm['X_emb'] = adata_code.X

    normalized_calibrated_target = torch.tensor(
        min_max_scaler_src.inverse_transform(calibrated_target.detach().numpy()))
    normalized_target = torch.tensor(min_max_scaler_target.inverse_transform(target.detach().numpy()))
    normalized_src = torch.tensor(min_max_scaler_src.inverse_transform(src.detach().numpy()))
    normalized_calibrated_src = torch.tensor(
        min_max_scaler_target.inverse_transform(calibrated_src.detach().numpy()))

    adata_target_calibrated_src = make_adata_from_batches(normalized_target.detach().numpy(),
                                                          normalized_calibrated_src.detach().numpy(), target_labels,
                                                          source_labels)

    adata_src_calibrated_target = make_adata_from_batches(normalized_src.detach().numpy(),
                                                          normalized_calibrated_target.detach().numpy(),
                                                          source_labels,
                                                          target_labels)

    if return_in == 'code_space':
        return adata_code

    elif return_in == 'original_space':
        # scale to original space

        return adata_src_calibrated_target, adata_target_calibrated_src

    elif return_in == 'original_space_and_code':
        # scale to original space

        return adata_code, adata_src_calibrated_target, adata_target_calibrated_src
