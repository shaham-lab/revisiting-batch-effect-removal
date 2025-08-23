# Batch Effect Removal Documentation Summary

This document provides a comprehensive overview of all functions and classes that have been documented with detailed docstrings in the batch effect removal codebase.

## SupervisedBer Module

### Core Network Architecture (`SupervisedBer/domain_adaption_net.py`)

#### Functions:
- **`init_weights(m)`**: Initialize weights of neural network layers with orthogonal initialization
- **`Net` class**: Neural network for supervised domain adaptation with batch effect removal
  - `__init__()`: Initialize the neural network architecture
  - `forward(x)`: Forward pass through the network

### Training Functions (`SupervisedBer/train_sda.py`)

#### Functions:
- **`get_absulute_gradient(net)`**: Calculate the L2 norm of gradients for all parameters in the network
- **`get_mutal_information(batch_y, mask0, mask1)`**: Find cells that exist in both batches based on their labels
- **`compute_class_weights(labels)`**: Compute class weights for imbalanced classification
- **`train_sda(...)`**: Train the supervised domain adaptation network
- **`get_count_class(images, nclasses)`**: Count the number of samples per class
- **`make_weights_for_balanced_classes(images, nclasses)`**: Create sample weights for balanced class sampling
- **`cdca_alignment(config, adata1, adata2, number_to_label, embed='', resample_data="resample")`**: Perform supervised domain adaptation using CDCA

### Utility Functions (`SupervisedBer/sda_utils.py`)

#### Functions:
- **`get_scale_fac(d, discard_diag=True)`**: Calculate scale factor for kernel functions based on pairwise distances
- **`get_weight_matrix(src_features, target_features)`**: Compute attention weight matrix between source and target features
- **`get_one_hot_encoding(labels, n_classes)`**: Convert integer labels to one-hot encoded format
- **`ccsa_loss(x, y, class_eq)`**: Compute Contrastive Semantic Alignment (CCSA) loss
- **`dsne_loss(src_feature, src_labels, tgt_feature, target_labels)`**: Compute Deep Siamese Network Embedding (dSNE) loss
- **`t_kernel(values)`**: Apply t-kernel transformation to distance values
- **`gaussian_kernel(values, scale)`**: Apply Gaussian kernel transformation to distance values

### Plotting Functions (`SupervisedBer/plot_data.py`)

#### Functions:
- **`plot_scatter(src_pca, target_pca, labels_b1, labels_b2, plot_dir, title='before-calibrationp')`**: Create scatter plots comparing source and target batches
- **`scatterHist(x1, x2, y1, y2, l1, l2, axis1='', axis2='', title='', name1='', name2='', plots_dir='', to_plot_labels=True)`**: Create a scatter plot with marginal histograms

## UnsupervisedBer Module

### Main Training Functions (`UnsupervisedBer/main.py`)

#### Functions:
- **`get_data_calibrated(src_data, target_data, encoder, decoder)`**: Generate calibrated data by swapping batch indicators during decoding
- **`validate(src, target, encoder, decoder)`**: Validate batch alignment using MMD (Maximum Mean Discrepancy)
- **`get_mutal_information(batch_y, mask0, mask1)`**: Find cells that exist in both batches based on their labels
- **`train2(src, target, data_loader, net, ind_discriminator, ae_optim, ind_disc_optim, config, dataset)`**: Train the unsupervised batch effect removal network
- **`ber_for_notebook(config, adata1, adata2, embed='', load_pre_weights='', return_in='original_space')`**: Main function for unsupervised batch effect removal

### Neural Network Components (`UnsupervisedBer/unsupervised/autoencoder.py`)

#### Classes:
- **`GeneratorModulation` class**: Generator modulation layer for style-based generation
  - `__init__(styledim, outch)`: Initialize the generator modulation layer
  - `forward(x, style)`: Apply style modulation to input features

- **`EqualLinear` class**: Equalized linear layer for stable training
  - `__init__(in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None)`: Initialize the equalized linear layer
  - `forward(input)`: Forward pass with equalized learning rate scaling

- **`Encoder` class**: Encoder network for unsupervised batch effect removal
  - `__init__(input_dim, hidden_dim=30, drop_prob=0.1, code_snape=25, hidden_layers=5, norm_layer=nn.BatchNorm1d, activation=ACTIVATION)`: Initialize the encoder architecture
  - `forward(x)`: Forward pass through the encoder
  - `save(path)`: Save encoder weights to file

- **`Decoder` class**: Decoder network for unsupervised batch effect removal
  - `__init__(code_dim, hidden_dim=20, output_dim=25, drop_prob=0.1, hidden_layers=10, norm_layer=nn.BatchNorm1d, activation=ACTIVATION)`: Initialize the decoder architecture
  - `forward(x)`: Forward pass through the decoder

#### Functions:
- **`init_weights(m)`**: Initialize weights of neural network layers

### Utility Functions (`UnsupervisedBer/unsupervised/utils.py`)

#### Functions:
- **`lr_scheduler(loss, ideal_loss, x_min=0.1, x_max=0.1, h_min=0.1, f_max=2.0)`**: Gap-aware Learning Rate Scheduler for Adversarial Networks
- **`compute_ind_disc_R1_loss(model, x, y)`**: Compute R1 regularization loss for independence discriminator
- **`indep_loss(logits, y, should_be_dependent=True)`**: Compute independence loss for the discriminator
- **`gradient_penalty(real, fake, f)`**: Compute gradient penalty for Wasserstein GAN training
- **`eval_mmd(source, target, num_pts=500)`**: Evaluate Maximum Mean Discrepancy (MMD) between source and target distributions
- **`get_cdca_term(src_feature, tgt_feature, src_label, tgt_label)`**: Compute Cross-Domain Cross-Attention (CDCA) terms

### Metrics and Evaluation (`UnsupervisedBer/metrics.py`)

#### Classes:
- **`MMD` class**: Maximum Mean Discrepancy (MMD) implementation for measuring distribution differences
  - `__init__(src, target, target_sample_size=1000, n_neighbors=500, scales=None, weights=None)`: Initialize the MMD module
  - `RaphyKernel(X, Y)`: Compute Raphy kernel between two sets of points
  - `cost()`: Calculate the MMD cost between source and target distributions

#### Functions:
- **`eval_mmd(source, target)`**: Evaluate MMD between source and target distributions

### Data Utilities (`UnsupervisedBer/data_utils.py`)

#### Functions:
- **`resample_data(data, sample_size=1000)`**: Resample data using distance-based weighted interpolation
- **`resample_data_normal(data, sample_size=1000)`**: Resample data using simple neighbor interpolation

### Dataset Classes (`UnsupervisedBer/unsupervised_dataset.py`)

#### Functions:
- **`organize_data(src_data, dest_data, src_labels=None, target_labels=None)`**: Organize source and destination data into a unified dataset format

#### Classes:
- **`UnsupervisedDataset` class**: PyTorch Dataset for unsupervised batch effect removal
  - `__init__(src_data, dest_data, src_labels=None, target_labels=None)`: Initialize the unsupervised dataset
  - `__len__()`: Get the total number of samples in the dataset
  - `__getitem__(idx)`: Get a single sample from the dataset

### Component Initialization (`UnsupervisedBer/get_component_config.py`)

#### Functions:
- **`initialize_components(config)`**: Initialize all neural network components for unsupervised batch effect removal

## Documentation Standards

All docstrings follow the Google Python Style Guide format and include:

1. **Brief description**: What the function/class does
2. **Detailed explanation**: How it works and its purpose in the batch effect removal pipeline
3. **Args section**: All parameters with types and descriptions
4. **Returns section**: Return values with types and descriptions
5. **Attributes section** (for classes): Class attributes with descriptions

## Key Concepts Documented

### Supervised Methods:
- Domain adaptation with labeled cell types
- Cross-Domain Cell-type Alignment (CDCA)
- Various loss functions (MMD, dSNE, CCSA)
- Class weight balancing for imbalanced data

### Unsupervised Methods:
- Autoencoder with batch-conditioned decoder
- Independence discriminator for batch-invariant representations
- Maximum Mean Discrepancy (MMD) for distribution alignment
- Adaptive learning rate scheduling for adversarial training

### Common Utilities:
- Data preprocessing and normalization
- Visualization and plotting functions
- Evaluation metrics and validation
- Neural network initialization and training

This documentation provides comprehensive coverage of the entire batch effect removal pipeline, making the codebase more accessible and maintainable for researchers and developers working in single-cell genomics.
