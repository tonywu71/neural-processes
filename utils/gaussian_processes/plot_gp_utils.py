from typing import Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import ConditionalNeuralProcess


def plot_mean_with_std(x: np.ndarray,
                       mean: np.ndarray,
                       std: np.ndarray,
                       y_true: Optional[np.ndarray]=None,
                       ax: Optional[plt.Axes]=None,
                       alpha: float=0.3) -> plt.Axes:
    """Plot mean and standard deviation.
    IMPORTANT: x should be sorted and mean and std should be sorted in the same order as x."""
    
    if ax is None:
        ax = plt.gca()
    
    # Plot mean:
    ax.plot(x, mean, label="Prediction")  # type: ignore
    
    # Plot target if given:
    if y_true is not None:
        ax.plot(x, y_true, label="True")
    
    # Plot standard deviation:
    ax.fill_between(x,  # type: ignore
                    mean - std,  # type: ignore
                    mean + std,  # type: ignore
                    alpha=alpha,
                    label="Standard deviation")
        
    return ax  # type: ignore


def plot_preds_from_ds_test(model: ConditionalNeuralProcess, ds_test: tf.data.Dataset, num_samples: int=1):
    fig, axis = plt.subplots(num_samples, 1, figsize=(num_samples*4, 5))
    
    (context_x, context_y, target_x), target_y = next(iter(ds_test.take(1)))
    y_preds = model.predict((context_x, context_y, target_x))
    
    batch_size = target_x.shape[0]
    assert num_samples <= batch_size, "num_samples must be smaller than batch_size"
    
    for batch_idx in range(num_samples):
        x_val = tf.squeeze(target_x[batch_idx, :, :])
        y_pred = tf.squeeze(y_preds[batch_idx, :, :])
        y_true = tf.squeeze(target_y[batch_idx, :, :])
        
        idx_x_sorted = tf.argsort(x_val)
        
        x_val = tf.gather(x_val, idx_x_sorted)
        mean = tf.gather(y_pred[:, 0], idx_x_sorted)
        std = tf.gather(y_pred[:, 1], idx_x_sorted)
        
        axis[batch_idx] = plot_mean_with_std(x=x_val, mean=mean, std=std)
    
    fig.suptitle("Predictions from test set")
    
    return fig
