import torch

from unsupervised.autoencoder import Encoder
from unsupervised.ber_network import DecoderBer, Net
from unsupervised.ind_discrimnator import IndDiscriminator


# Assuming Encoder, DecoderBer, Net, and IndDiscriminator classes are defined elsewhere

def initialize_components(config):
    """
    Initialize all neural network components for unsupervised batch effect removal.
    
    This function creates and configures the encoder, decoder, independence discriminator,
    and optimizers based on the provided configuration dictionary. It sets up the
    complete neural network architecture needed for unsupervised batch effect removal.
    
    Args:
        config (dict): Configuration dictionary containing the following keys:
            - input_dim (int): Dimension of input features (number of genes)
            - hidden_dim_encoder (int): Hidden dimension for encoder
            - hidden_dim_decoder (int): Hidden dimension for decoder
            - code_dim (int): Dimension of latent code
            - drop_prob (float): Dropout probability
            - lr (float): Learning rate for optimizers
            - weight_decay (float): Weight decay for optimizers
    
    Returns:
        tuple: (net, ind_discriminator, ae_optim, ind_disc_optim)
            - net (Net): Combined encoder-decoder network
            - ind_discriminator (IndDiscriminator): Independence discriminator
            - ae_optim (torch.optim.Adam): Optimizer for autoencoder
            - ind_disc_optim (torch.optim.SGD): Optimizer for independence discriminator
    """
    encoder = Encoder(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim_encoder"],
        drop_prob=config["drop_prob"],
        code_snape=config["code_dim"]
    )

    decoder = DecoderBer(
        code_dim=config["code_dim"],
        hidden_dim=config["hidden_dim_decoder"],
        output_dim=config["input_dim"],
        drop_prob=config["drop_prob"]
    )

    net = Net(encoder, decoder)

    ind_discriminator = IndDiscriminator(
        input_dim=config["code_dim"],
        hidden_dim=config["hidden_dim_encoder"],  # Assuming hidden_dim_encoder is used here as well
        drop_prob=config["drop_prob"]
    )

    ae_optim = torch.optim.Adam(
        net.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    ind_disc_optim = torch.optim.SGD(
        ind_discriminator.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    return net, ind_discriminator, ae_optim, ind_disc_optim


# Usage
# net, ind_discriminator, ae_optim, ind_disc_optim = initialize_components(config)
