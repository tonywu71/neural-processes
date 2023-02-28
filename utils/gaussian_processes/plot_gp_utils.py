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
    ax.plot(x, mean, label="Mean prediction")  # type: ignore
    
    # Plot target if given:
    if y_true is not None:
        ax.plot(x, y_true, label="True function")
    
    # Plot standard deviation:
    ax.fill_between(x,  # type: ignore
                    mean - std,  # type: ignore
                    mean + std,  # type: ignore
                    alpha=alpha,
                    label="Standard deviation")
    
    ax.legend()
    
    return ax  # type: ignore


def plot_preds_from_ds_test(model: ConditionalNeuralProcess,
                            ds_test: tf.data.Dataset,
                            num_samples: int=1,
                            show_context_points: bool=True):
    fig, axis = plt.subplots(num_samples, 1, figsize=(8, num_samples*2))
    
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
        y_true = tf.gather(y_true, idx_x_sorted)
        
        if num_samples == 1:
            plot_mean_with_std(x=x_val, mean=mean, std=std, y_true=y_true, ax=axis)
            if show_context_points:
                axis.scatter(context_x[batch_idx, :, :], context_y[batch_idx, :, :])  # type: ignore
        else:
            plot_mean_with_std(x=x_val, mean=mean, std=std, y_true=y_true, ax=axis[batch_idx])
            if show_context_points:
                axis[batch_idx].scatter(context_x[batch_idx, :, :], context_y[batch_idx, :, :])
    
    fig.suptitle("Predictions from test set")
    
    fig.tight_layout()
    
    return fig


def plot_preds_from_single_example(model: ConditionalNeuralProcess,
                                   context_x: tf.Tensor,
                                   context_y: tf.Tensor,
                                   target_x: tf.Tensor,
                                   target_y: tf.Tensor,
                                   show_context_points: bool=True,
                                   ax: Optional[plt.Axes]=None) -> plt.Axes:
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # Add batch dimension:
    context_x = tf.expand_dims(context_x, axis=0)
    context_y = tf.expand_dims(context_y, axis=0)
    target_x = tf.expand_dims(target_x, axis=0)
    target_y = tf.expand_dims(target_y, axis=0)
    
    y_preds = model.predict((context_x, context_y, target_x))
    
    x_val = tf.squeeze(target_x)
    y_pred = tf.squeeze(y_preds)
    y_true = tf.squeeze(target_y)
    
    idx_x_sorted = tf.argsort(x_val)
    
    x_val = tf.gather(x_val, idx_x_sorted)
    mean = tf.gather(y_pred[:, 0], idx_x_sorted)
    std = tf.gather(y_pred[:, 1], idx_x_sorted)
    y_true = tf.gather(y_true, idx_x_sorted)
    
    plot_mean_with_std(x=x_val, mean=mean, std=std, y_true=y_true, ax=ax)
    if show_context_points:
        ax.scatter(context_x, context_y)  # type: ignore
    
    num_samples = context_x.shape[1]
    ax.set_title(f"{num_samples = }")
    
    return ax
