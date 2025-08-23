import math
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ACTIVATION = nn.ReLU()


class GeneratorModulation(torch.nn.Module):
    """
    Generator modulation layer for style-based generation.
    
    This module applies adaptive instance normalization (AdaIN) style modulation
    to input features using style vectors. It's commonly used in style-based
    generative models.
    
    Attributes:
        scale (EqualLinear): Linear layer for scale modulation
        bias (EqualLinear): Linear layer for bias modulation
    """
    
    def __init__(self, styledim, outch):
        """
        Initialize the generator modulation layer.
        
        Args:
            styledim (int): Dimension of style vector
            outch (int): Number of output channels
        """
        super().__init__()
        self.scale = EqualLinear(styledim, outch)
        self.bias = EqualLinear(styledim, outch)

    def forward(self, x, style):
        """
        Apply style modulation to input features.
        
        Args:
            x (torch.Tensor): Input features
            style (torch.Tensor): Style vector
        
        Returns:
            torch.Tensor: Modulated features
        """
        return x * (1 * self.scale(style)) + self.bias(style)


class EqualLinear(nn.Module):
    """
    Equalized linear layer for stable training.
    
    This layer uses equalized learning rate scaling to maintain stable training
    in deep networks. It's commonly used in progressive GAN architectures.
    
    Attributes:
        weight (nn.Parameter): Learnable weight matrix
        bias (nn.Parameter): Learnable bias vector
        activation (nn.Module): Activation function
        scale (float): Scaling factor for equalized learning rate
        lr_mul (float): Learning rate multiplier
    """
    
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        """
        Initialize the equalized linear layer.
        
        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            bias (bool): Whether to include bias term
            bias_init (float): Initial bias value
            lr_mul (float): Learning rate multiplier
            activation (nn.Module): Activation function
        """
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        """
        Forward pass with equalized learning rate scaling.
        
        Args:
            input (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output tensor
        """
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


def init_weights(m):
    """
    Initialize weights of neural network layers.
    
    This function applies Xavier uniform initialization to linear layers and
    constant initialization to batch normalization layers. It also sets random
    seeds for reproducibility.
    
    Args:
        m (nn.Module): A PyTorch module (typically a layer)
    
    Returns:
        None: Modifies the module in-place
    """
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        # nn.init.constant_(m.bias, 0.01)

    elif type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    """
    Encoder network for unsupervised batch effect removal.
    
    This encoder transforms input gene expression data into a lower-dimensional
    latent representation. It uses residual connections and batch normalization
    for stable training.
    
    Attributes:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden layers
        drop_prob (float): Dropout probability
        norm_layer (nn.Module): Normalization layer type
        activation (nn.Module): Activation function
        hidden_layers (int): Number of hidden layers
        hidden_layers_list (list): List of hidden layer modules
        input (nn.Sequential): Input processing layers
        code (nn.Sequential): Code generation layers
    """
    
    def __init__(self,
                 input_dim,
                 hidden_dim=30,
                 drop_prob=0.1,
                 code_snape=25,
                 hidden_layers=5,
                 norm_layer=nn.BatchNorm1d,
                 activation=ACTIVATION):
        """
        Initialize the encoder architecture.
        
        Args:
            input_dim (int): Dimension of input features (number of genes)
            hidden_dim (int): Dimension of hidden layers
            drop_prob (float): Dropout probability for regularization
            code_snape (int): Dimension of latent code (typo in original, should be code_shape)
            hidden_layers (int): Number of hidden layers
            norm_layer (nn.Module): Type of normalization layer
            activation (nn.Module): Activation function
        """
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.norm_layer = norm_layer
        self.activation = activation
        self.hidden_layers_list = []
        use_bias = norm_layer != nn.BatchNorm1d
        self.hidden_layers = hidden_layers
        # Layers
        self.input = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=use_bias),
            self.norm_layer(self.hidden_dim),
            self.activation,
            nn.Dropout(self.drop_prob)
        )
        self.input.apply(init_weights)
        for i in range(self.hidden_layers):
            layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=use_bias),
                self.norm_layer(self.hidden_dim),
                self.activation,
                nn.Dropout(drop_prob)
            )
            layer.apply(init_weights)
            self.hidden_layers_list.append(layer)

        self.code = nn.Sequential(
            nn.Linear(self.hidden_dim, code_snape)
        )
        self.code.apply(init_weights)

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Performs forward propagation through the input layer, hidden layers with
        residual connections, and code generation layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Latent code representation
        """
        x = self.input(x)

        for hidden_layer in self.hidden_layers_list:
            x = x + hidden_layer(x)

        return self.code(x)

    def save(self, path):
        """
        Save encoder weights to file.
        
        Args:
            path (str): Directory path to save weights
        """
        state_dict = self.state_dict()
        weights_path = os.path.join(path, "encoder.pt")
        torch.save(state_dict, weights_path)


class Decoder(nn.Module):
    """
    Decoder network for unsupervised batch effect removal.
    
    This decoder reconstructs gene expression data from latent representations.
    It uses residual connections and batch normalization for stable training.
    
    Attributes:
        output_dim (int): Dimension of output features
        hidden_dim (int): Dimension of hidden layers
        drop_prob (float): Dropout probability
        norm_layer (nn.Module): Normalization layer type
        activation (nn.Module): Activation function
        hidden_layers (int): Number of hidden layers
        hidden_layers_list (list): List of hidden layer modules
        input (nn.Sequential): Input processing layers
        last_layer (nn.Sequential): Output generation layers
    """
    
    def __init__(self,
                 code_dim,
                 hidden_dim=20,
                 output_dim=25,
                 drop_prob=0.1,
                 hidden_layers=10,
                 norm_layer=nn.BatchNorm1d,
                 activation=ACTIVATION,
                 ):
        """
        Initialize the decoder architecture.
        
        Args:
            code_dim (int): Dimension of input latent code
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output features (number of genes)
            drop_prob (float): Dropout probability for regularization
            hidden_layers (int): Number of hidden layers
            norm_layer (nn.Module): Type of normalization layer
            activation (nn.Module): Activation function
        """
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.norm_layer = norm_layer
        self.activation = activation
        self.hidden_layers = hidden_layers
        self.hidden_layers_list = []
        use_bias = norm_layer != nn.BatchNorm1d
        # Layers
        self.input = nn.Sequential(
            nn.Linear(code_dim, self.hidden_dim),
            self.norm_layer(self.hidden_dim),
            self.activation,
            nn.Dropout(self.drop_prob)
        )
        self.input.apply(init_weights)
        for i in range(self.hidden_layers):
            layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=use_bias),
                self.norm_layer(self.hidden_dim),
                self.activation,
                nn.Dropout(drop_prob)
            )
            layer.apply(init_weights)
            self.hidden_layers_list.append(layer)

        self.last_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.last_layer.apply(init_weights)

    def forward(self, x):
        """
        Forward pass through the decoder.
        
        Performs forward propagation through the input layer, hidden layers with
        residual connections, and output generation layer.
        
        Args:
            x (torch.Tensor): Input latent code of shape (batch_size, code_dim)
        
        Returns:
            torch.Tensor: Reconstructed gene expression data
        """
        x = self.input(x)

        for hidden_layer in self.hidden_layers_list:
            x = x + hidden_layer(x)

        return self.last_layer(x)
