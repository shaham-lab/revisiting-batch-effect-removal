import torch.nn as nn
from metrics import MMD

import torch
def lr_scheduler(loss, ideal_loss, x_min=0.1, x_max=0.1, h_min=0.1, f_max=2.0):
    """
    Gap-aware Learning Rate Scheduler for Adversarial Networks.

    This scheduler adjusts the learning rate based on the gap between the current
    discriminator loss and the ideal loss. It helps maintain stable training
    in adversarial settings by preventing the discriminator from becoming too
    strong or too weak.

    Args:
        loss (torch.Tensor): the loss of the discriminator D on the training data.
        ideal_loss (float): the ideal loss of D.
        x_min (float): the value of x at which the scheduler achieves its minimum allowed value h_min.
        x_max (float): the value of x at which the scheduler achieves its maximum allowed value f_max.
        h_min (float, optional): a scalar in (0, 1] denoting the minimum allowed value of the scheduling function.
                                Defaults to 0.1.
        f_max (float, optional): a scalar (>= 1) denoting the maximum allowed value of the scheduling function.
                                Defaults to 2.0.

    Returns:
        torch.Tensor: A scalar in [h_min, f_max], which can be used as a multiplier for the learning rate of D.
    """

    x = torch.abs(torch.tensor(loss - ideal_loss))
    f_x = torch.clamp(torch.pow(f_max, x / x_max), 1.0, f_max)
    h_x = torch.clamp(torch.pow(h_min, x / x_min), h_min, 1.0)

    # Use a conditional statement to handle the loss comparison
    if loss >= ideal_loss:
        return f_x
    else:
        return h_x


def compute_ind_disc_R1_loss(model, x, y):
    """
    Compute R1 regularization loss for independence discriminator.
    
    R1 regularization penalizes the gradient norm of the discriminator with respect
    to real data, helping to stabilize training and prevent mode collapse.
    
    Args:
        model (nn.Module): Independence discriminator model
        x (torch.Tensor): Input data
        y (torch.Tensor): Batch indicators
    
    Returns:
        torch.Tensor: R1 regularization loss
    """
    x = x.detach().clone()
    x.requires_grad_()
    pred_real = model(x).sum()
    grad_real, = torch.autograd.grad(
        outputs=pred_real,
        inputs=[x],
        create_graph=True,
        retain_graph=True,
    )
    grad_real2 = grad_real.pow(2)
    dims = list(range(1, grad_real2.ndim))
    grad_penalty = grad_real2.sum(dims) * 0.5

    return grad_penalty.sum()

import torch.nn.functional as F

ind_criterion = F.binary_cross_entropy


def indep_loss(logits, y, should_be_dependent=True):
    """
    Compute independence loss for the discriminator.
    
    This function computes the binary cross-entropy loss between the discriminator
    predictions and batch indicators. When should_be_dependent=True, it encourages
    the discriminator to predict batch membership correctly. When should_be_dependent=False,
    it encourages the discriminator to be unable to predict batch membership.
    
    Args:
        logits (torch.Tensor): Discriminator predictions
        y (torch.Tensor): Batch indicators (0 or 1)
        should_be_dependent (bool): Whether the discriminator should be able to predict batch membership
    
    Returns:
        torch.Tensor: Independence loss value
    """
    logits = logits.squeeze()
    y = y.squeeze()
    y = torch.clamp(y, min=1e-3, max=1 - 1e-3)
    logits = torch.clamp(logits, min=1e-3, max=1 - 1e-3)
    # print(logits.min(), logits.max())

    if should_be_dependent:
        return ind_criterion(logits, y)
    else:
        return -ind_criterion(logits, y)


import torch


def gradient_penalty(real, fake, f):
    """
    Compute gradient penalty for Wasserstein GAN training.
    
    This function computes the gradient penalty term used in Wasserstein GAN with
    gradient penalty (WGAN-GP) to enforce the 1-Lipschitz constraint on the discriminator.
    
    Args:
        real (torch.Tensor): Real data samples
        fake (torch.Tensor): Fake data samples
        f (nn.Module): Discriminator function
    
    Returns:
        torch.Tensor: Gradient penalty loss
    """
    def interpolate(a, b):
        shape = [a.size(0)] + [1] * (a.dim() - 1)
        alpha = torch.rand(shape)
        b_resized = torch.nn.functional.interpolate(b.unsqueeze(0).unsqueeze(0), size=a.shape, mode='bilinear',
                                                    align_corners=False).squeeze()
        inter = a + alpha * (b_resized - a)
        return inter

    x = interpolate(real, fake)
    pred = f(x)
    gradients = torch.autograd.grad(outputs=pred, inputs=x,
                                    grad_outputs=torch.ones_like(pred),
                                    create_graph=True, retain_graph=True)[0]
    slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1))
    gp = torch.mean((slopes - 1.) ** 2)
    return gp



def eval_mmd(source, target, num_pts=500):
    """
    Evaluate Maximum Mean Discrepancy (MMD) between source and target distributions.
    
    MMD is a kernel-based method for measuring the difference between two probability
    distributions. It's commonly used in domain adaptation to measure how well
    two batches have been aligned.
    
    Args:
        source (torch.Tensor): Source distribution samples
        target (torch.Tensor): Target distribution samples
        num_pts (int): Number of points to sample for MMD computation
    
    Returns:
        tuple: (mmd_mean, mmd_std) - Mean and standard deviation of MMD estimates
    """
    return MMD(source, target).cost()


def get_cdca_term(src_feature, tgt_feature, src_label, tgt_label):
    """
    Compute Cross-Domain Cross-Attention (CDCA) terms.
    
    This function computes attention matrices between source and target features
    and labels for cross-domain alignment. The attention matrices help identify
    corresponding regions in the feature space across different batches.
    
    Args:
        src_feature (torch.Tensor): Source batch features
        tgt_feature (torch.Tensor): Target batch features
        src_label (torch.Tensor): Source batch labels
        tgt_label (torch.Tensor): Target batch labels
    
    Returns:
        tuple: (attention_s_t, attention_t_s, attention_s_s, attention_t_t)
            - attention_s_t: Attention from source to target features
            - attention_t_s: Attention from target to source features
            - attention_s_s: Self-attention for source features
            - attention_t_t: Self-attention for target features
    """
    attention_s_t = torch.nn.functional.softmax(src_feature @ tgt_feature.t()/2, dim=-1)
    attention_t_s = torch.nn.functional.softmax(tgt_feature @ src_feature.t()/2, dim=-1)
    attention_s_s = torch.nn.functional.softmax(src_feature @ src_feature.t()/2, dim=-1)
    attention_t_t = torch.nn.functional.softmax(tgt_feature @ tgt_feature.t()/2, dim=-1)


    labels_t_s = attention_t_s @ src_label
    labels_s_t = attention_s_t @ tgt_label
    labels_s_s = attention_s_s @ src_label
    labels_t_t = attention_t_t @ tgt_label

    labels_t_s = torch.clip(labels_t_s,1e-3,1-1e-3)
    labels_s_t = torch.clip(labels_s_t,1e-3,1-1e-3)
    labels_s_s = torch.clip(labels_s_s,1e-3,1-1e-3)
    labels_t_t = torch.clip_(labels_t_t,1e-3,1-1e-3)


    return labels_t_s ,labels_s_t,labels_s_s,labels_t_t
