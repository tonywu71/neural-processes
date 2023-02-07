from functools import partial
from typing import Tuple, Callable, Iterator

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

tfd = tfp.distributions

MIN_NUM_CONTEXT = 3
MIN_NUM_TARGET = 2


def gen(x_values, gp, batch_size, num_context, num_target, iterations, testing) -> Iterator[Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]]:        
    for _ in range(iterations):
        if testing is True:
            num_target = 400
            num_total_points = num_target
            x_values = tf.expand_dims(tf.range(-2., 2., 1./100., dtype=tf.float32), axis=0)  # (1, 400)
            x_values = tf.tile(x_values, [batch_size, 1])  # (batch_size, 400)
            x_values = tf.expand_dims(x_values, axis=-1)  # (batch_size, 400, 1)
        else:
            x_values = x_values
            num_target = num_target
        
        y_values = tf.expand_dims(gp.sample(), axis=-1)

        if testing is True:
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = tf.random.shuffle(tf.range(num_target))
            context_x = tf.gather(x_values, idx[:num_context], axis=1)
            context_y = tf.gather(y_values, idx[:num_context], axis=1)
        
        else:
            # Select the targets which will consist of the context points
            # as well as some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        yield (context_x, context_y, target_x), target_y


def get_gp_curve_generator(x_values: np.ndarray,
                           gp: tfp.distributions.Distribution,
                           num_context: int,
                           num_target: int,
                           iterations: int=10000,
                           testing: bool=False) -> Callable:
    
    batch_size, num_total_points, _ = x_values.shape
    
    return partial(gen,
                   x_values=x_values,
                   gp=gp,
                   batch_size=batch_size,
                   num_context=num_context,
                   num_target=num_target,
                   iterations=iterations,
                   testing=testing)


def get_gp_curve_generator_from_uniform(iterations: int=10000,
                                        batch_size: int=64,
                                        max_num_context: int=10,
                                        kernel_length_scale: float=0.4,
                                        testing: bool=False) -> Callable:
    num_context = tf.random.uniform(shape=[],
                                minval=MIN_NUM_CONTEXT,
                                maxval=max_num_context,
                                dtype=tf.int32)

    num_target = tf.random.uniform(shape=[],
                                minval=MIN_NUM_TARGET,
                                maxval=max_num_context,
                                dtype=tf.int32)
    
    num_total_points = num_context + num_target
    
    x_values = tf.random.uniform(shape=(batch_size, num_total_points, 1),
                                    minval=-2,
                                    maxval=2)
    
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=kernel_length_scale)
    
    gp = tfd.GaussianProcess(kernel, index_points=x_values, jitter=1.0e-4)
    
    return get_gp_curve_generator(x_values=x_values,
                                  gp=gp,
                                  num_context=num_context,
                                  num_target=num_target,
                                  iterations=iterations,
                                  testing=testing)


def load_regression_data(iterations: int=250, batch_size: int=32) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.data.Dataset.from_generator(
        get_gp_curve_generator_from_uniform(
            iterations=iterations,
            batch_size=batch_size,
            max_num_context=10,
            testing=False),
        output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
    )
    test_ds = tf.data.Dataset.from_generator(
        get_gp_curve_generator_from_uniform(
            iterations=iterations,
            batch_size=batch_size,
            testing=True),
        output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
    )
    
    train_ds = train_ds.shuffle(buffer_size=10).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_ds, test_ds



class DataGenerator():
    """Class that uses load_regression_data to create datasets.
    """
    def __init__(self, iterations: int=250, batch_size: int=32):
        self.iterations = iterations
        self.batch_size = batch_size
        
        self.train_ds, self.test_ds = load_regression_data(
            iterations=iterations, batch_size=batch_size)
    
    def plot_random_batch(self, figsize=(8, 5)):
        """Plot a random batch from the train_ds.
        """
        (context_x, context_y, target_x), target_y = next(iter(self.train_ds.take(1)))
        context_x = context_x.numpy()
        context_y = context_y.numpy()
        target_x = target_x.numpy()
        target_y = target_y.numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(target_x[0, :, 0], target_y[0, :, 0], c="blue", label='Target')
        ax.scatter(context_x[0, :, 0], context_y[0, :, 0], marker="x", c="red", label='Observations')
        ax.legend()

        return fig, ax
