from typing import Optional
from abc import ABC

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

import matplotlib.pyplot as plt


class RegressionDataGeneratorBase(ABC):
    """Class that uses load_regression_data to create datasets.
    """
    def __init__(self, iterations: int=250, batch_size: int=32):
        self.iterations = iterations
        self.batch_size = batch_size
        
        self.train_ds: Optional[tf.data.Dataset] = None
        self.test_ds: Optional[tf.data.Dataset] = None
    
    
    @staticmethod
    def plot_batch(context_x, context_y, target_x, target_y, figsize=(8, 5)):
        context_x = context_x.numpy()
        context_y = context_y.numpy()
        target_x = target_x.numpy()
        target_y = target_y.numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(target_x[0, :, 0], target_y[0, :, 0], c="blue", label='Target')
        ax.scatter(context_x[0, :, 0], context_y[0, :, 0], marker="x", c="red", label='Observations')
        ax.legend()

        return fig, ax
    
    
    def plot_random_batch(self, figsize=(8, 5)):
        """Plot a random batch from the train_ds.
        """
        (context_x, context_y, target_x), target_y = next(iter(self.train_ds.take(1)))
        fig, ax = RegressionDataGeneratorBase.plot_batch(context_x, context_y, target_x, target_y, figsize=figsize)
        return fig, ax
