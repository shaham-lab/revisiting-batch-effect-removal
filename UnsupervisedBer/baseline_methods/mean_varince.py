import numpy as np
from sklearn.preprocessing import StandardScaler


def batch_effect_correction(batch_a, batch_b):
    """
    Correct batch effects by:
      1. Standardizing batch A (Z-score normalization).
      2. Applying batch B's inverse transform to the standardized batch A.

    This will adjust batch A to have the mean and standard deviation of batch B.

    Args:
        batch_a (np.ndarray): Data from batch A (samples x features).
        batch_b (np.ndarray): Data from batch B (samples x features).

    Returns:
        np.ndarray: Transformed batch A with the scale of batch B.
    """
    # Standardize batch A using its own statistics.
    scaler_a = StandardScaler()
    batch_a_z = scaler_a.fit_transform(batch_a)

    # Fit a scaler on batch B to capture its mean and std.
    scaler_b = StandardScaler()
    scaler_b.fit(batch_b)

    # Apply the inverse transform of batch B's scaler on batch A's z-scores.
    # This maps the z-scores to batch B's original scale.
    batch_a_corrected = scaler_b.inverse_transform(batch_a_z)

    return batch_a_corrected


# Example usage:
if __name__ == '__main__':
    np.random.seed(42)
    # Simulate some data: Batch A with mean 5, std 2; Batch B with mean 10, std 3.
    batch_a = np.random.normal(5, 2, (100, 20))
    batch_b = np.random.normal(10, 3, (100, 20))

    corrected_batch_a = batch_effect_correction(batch_a, batch_b)

    # Display the means of the original and corrected batches.
    print("Batch A Mean (original):", np.mean(batch_a, axis=0))
    print("Batch A Mean (corrected):", np.mean(corrected_batch_a, axis=0))
    print("Batch B Mean:", np.mean(batch_b, axis=0))
