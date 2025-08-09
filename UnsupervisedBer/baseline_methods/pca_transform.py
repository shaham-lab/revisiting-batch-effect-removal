import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


def batch_correction_pca(data, batch_labels, n_components=20):
    """
    Performs batch correction by removing the PC most correlated with batch effects.

    Parameters:
    - data: array-like of shape (n_samples, n_features)
        The original data matrix.
    - batch_labels: array-like of shape (n_samples,)
        Batch assignments for each sample. (Should be numeric or encoded.)
    - n_components: int or None
        Number of principal components to compute. If None, all components are kept.

    Returns:
    - corrected_data: array-like of shape (n_samples, n_features)
        The reconstructed data with the batch-associated PC removed.
    - removed_pc: int
        The index (0-based) of the principal component that was removed.
    """
    # 1. Standardize the data (zero mean, unit variance)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 2. Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data_scaled)  # scores: (n_samples, n_components)
    components = pca.components_  # components: (n_components, n_features)

    # 3. Compute correlation of each PC with the batch variable
    #    We take the absolute value to assess strength regardless of direction.
    correlations = []
    for i in range(scores.shape[1]):
        # Compute Pearson correlation between the i-th PC score and the batch labels.
        # If batch_labels is categorical, consider encoding it to numeric.
        corr = np.abs(np.corrcoef(scores[:, i], batch_labels)[0, 1])
        correlations.append(corr)

    correlations = np.array(correlations)

    # 4. Identify the PC most correlated with the batch effect
    removed_pc = int(np.argmax(correlations))
    print(f"Removing PC {removed_pc} (correlation = {correlations[removed_pc]:.3f} with batch)")

    # 5. Remove the identified PC from the scores and components
    #    We remove the column (PC score) and row (component vector) corresponding to the removed PC.
    scores_filtered = np.delete(scores, removed_pc, axis=1)
    components_filtered = np.delete(components, removed_pc, axis=0)

    # 6. Reconstruct the data using the remaining PCs.
    #    Note: The PCA reconstruction (without the removed component) is given by:
    #          X_reconstructed = scores_filtered @ components_filtered
    data_reconstructed_scaled = np.dot(scores_filtered, components_filtered)

    # 7. Reverse the scaling transformation to return to the original data space.
    corrected_data = scaler.inverse_transform(data_reconstructed_scaled)

    return corrected_data, removed_pc


# Example usage:
if __name__ == "__main__":
    # Suppose we have a data matrix (e.g., gene expression) and batch labels.
    # For demonstration, we create a synthetic dataset.
    np.random.seed(42)
    n_samples, n_features = 100, 50
    # Simulate data with some batch effect:
    batch_labels = np.random.choice([0, 1], size=n_samples)  # two batches: 0 and 1
    # Create data: base signal + batch effect
    base_signal = np.random.randn(n_samples, n_features)
    batch_effect = np.outer(batch_labels, np.random.randn(n_features)) * 0.5
    data = base_signal + batch_effect

    # Perform batch correction using PCA.
    corrected_data, removed_pc = batch_correction_pca(data, batch_labels)

    # Compare the original and corrected data (this could include visualization or further analysis).
    print("Original data shape:", data.shape)
    print("Corrected data shape:", corrected_data.shape)
