import torch
from torch import nn


def init_weights(m):
    """
    Initialize weights of neural network layers with orthogonal initialization.
    
    This function applies orthogonal initialization to linear layers and constant
    initialization to batch normalization layers. Orthogonal initialization helps
    maintain gradient flow in deep networks.
    
    Args:
        m (nn.Module): A PyTorch module (typically a layer)
    
    Returns:
        None: Modifies the module in-place
    """
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight, gain=1)
        # m.bias.data.zero_()
    elif type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Net(nn.Module):
    """
    Neural network for supervised domain adaptation with batch effect removal.
    
    This network implements a deep neural network with residual connections,
    batch normalization, and dropout for learning shared representations
    across different batches while preserving cell type information.
    
    Attributes:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden layers
        drop_prob (float): Dropout probability
        norm_layer (nn.Module): Normalization layer type (e.g., BatchNorm1d)
        activation (nn.Module): Activation function
        hidden_layers (int): Number of hidden layers
        hidden_layers_list (nn.ModuleList): List of hidden layer modules
        features (nn.Sequential): Feature extraction layers
        head (nn.Sequential): Classification head with softmax output
    """
    
    def __init__(self, input_dim=25,
                 hidden_dim=20,
                 drop_prob=0.5,
                 hidden_layers=5,
                 norm_layer=nn.BatchNorm1d,
                 activation=nn.ReLU(),class_num=2):
        """
        Initialize the neural network architecture.
        
        Args:
            input_dim (int): Dimension of input features (number of genes)
            hidden_dim (int): Dimension of hidden layers
            drop_prob (float): Dropout probability for regularization
            hidden_layers (int): Number of hidden layers in the network
            norm_layer (nn.Module): Type of normalization layer to use
            activation (nn.Module): Activation function for hidden layers
            class_num (int): Number of output classes (cell types)
        """
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.norm_layer = norm_layer
        self.activation = activation
        self.hidden_layers = hidden_layers

        use_bias = norm_layer != nn.BatchNorm1d
        self.hidden_layers_list = []

        # Layers
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=use_bias),
            self.norm_layer(self.hidden_dim),
            self.activation,
        )
        self.input_layer.apply(init_weights)
        for i in range(self.hidden_layers):
            layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=use_bias),
                self.norm_layer(self.hidden_dim),
                self.activation,
                nn.Dropout(drop_prob)
            )
            layer.apply(init_weights)
            self.hidden_layers_list.append(layer)
        self.hidden_layers_list = nn.ModuleList(self.hidden_layers_list)

        self.features = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=use_bias),
            self.norm_layer(self.hidden_dim),
            self.activation,
            nn.Dropout(drop_prob)
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=use_bias),
            self.norm_layer(self.hidden_dim),
            self.activation,
            nn.Dropout(drop_prob),
            nn.Linear(self.hidden_dim, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Performs forward propagation through the input layer, hidden layers with
        residual connections, feature extraction, and classification head.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            tuple: (prediction, activations)
                - prediction (torch.Tensor): Softmax probabilities for cell type classification
                - activations (torch.Tensor): Feature representations from the feature layer
        """
        x = self.input_layer(x)

        for hidden_layer in self.hidden_layers_list:
            x = x + hidden_layer(x)

        activations = self.features(x)
        prediction = self.head(activations).squeeze()

        return prediction, activations

