from functools import partial
from typing import Optional, Tuple, Callable, Iterator

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from dataloader.regression_data_generator_base import RegressionDataGeneratorBase


def gen_from_arbitrary_gp(
        batch_size: int,
        iterations: int,
        min_kernel_length_scale: float,
        max_kernel_length_scale: float,
        min_num_context: int,
        max_num_context: int,
        min_num_target: int,
        max_num_target: int,
        min_x_val_uniform: float,
        max_x_val_uniform: float,
        testing: bool) -> Iterator[Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]]:
    """Generates a batch of data for regression based on the original Conditional Neural Processes paper.
    Note that the data is generated batch-wise.
    
    During training and for each batch:
    - Both num_context and num_target are drawn from uniform distributions
    - The (num_context + num_target) x_values are drawn from a uniform distribution
    - A Gaussian Process with predefined kernel and a null mean function is used to generate the y_values from the x_values
    """
    
    for _ in range(iterations):
        # Set kernel length scale:
        kernel_length_scale = tf.random.uniform(shape=[],
                                                minval=min_kernel_length_scale,  # type: ignore
                                                maxval=max_kernel_length_scale,
                                                dtype=tf.dtypes.float32)
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=kernel_length_scale)
        
        
        # NB: The distribution of y_values is the same for each iteration (i.e. the the one defined by
        #     the arbitrary GP) but the sampled x_values do differ (in terms of size and values).
        num_context = tf.random.uniform(shape=[],
                                        minval=min_num_context,
                                        maxval=max_num_context,
                                        dtype=tf.int32)

        if not testing:
            num_target = tf.random.uniform(shape=[],
                                           minval=min_num_target,
                                           maxval=max_num_target,
                                           dtype=tf.int32)
        else:
            # If testing, we want to use a fixed number of points for the target
            num_target = max_num_target - 1  # -1 because max_num_target is exclusive
        
        num_total_points = num_context + num_target
        
        x_values = tf.random.uniform(shape=(batch_size, num_total_points, 1),
                                     minval=min_x_val_uniform,
                                     maxval=max_x_val_uniform)
        
        gp = tfd.GaussianProcess(kernel, index_points=x_values, jitter=1.0e-4)
        y_values = tf.expand_dims(gp.sample(), axis=-1)
        
        idx = tf.random.shuffle(tf.range(num_total_points))
        
        # Select the targets which will consist of the context points
        # as well as some new target points
        target_x = x_values[:, :, :]
        target_y = y_values[:, :, :]

        # Select the observations
        context_x = tf.gather(x_values, indices=idx[:num_context], axis=1)
        context_y = tf.gather(y_values, indices=idx[:num_context], axis=1)
        
        if tf.shape(context_x) != tf.shape(context_y):
            continue
        if tf.shape(target_x) != tf.shape(target_y):
            continue
        if tf.shape(context_x)[-1] != tf.shape(target_x)[-1]:
            continue

        yield (context_x, context_y, target_x), target_y


class RegressionDataGeneratorArbitraryGPWithVaryingKernel(RegressionDataGeneratorBase):
    """Class that generates a batch of data for regression based on
    the original Conditional Neural Processes paper."""
    def __init__(self,
                 iterations: int=250,
                 batch_size: int=32,
                 min_num_context: int=3,
                 max_num_context: int=10,
                 min_num_target: int=2,
                 max_num_target: int=10,
                 min_x_val_uniform: int=-2,
                 max_x_val_uniform: int=2,
                 n_iterations_test: Optional[int]=None,
                 min_kernel_length_scale: float=0.1,
                 max_kernel_length_scale: float=1.):
        super().__init__(iterations=iterations,
                         batch_size=batch_size,
                         min_num_context=min_num_context,
                         max_num_context=max_num_context,
                         min_num_target=min_num_target,
                         max_num_target=max_num_target,
                         min_x_val_uniform=min_x_val_uniform,
                         max_x_val_uniform=max_x_val_uniform,
                         n_iterations_test=n_iterations_test)
        
        self.min_kernel_length_scale = min_kernel_length_scale
        self.max_kernel_length_scale = max_kernel_length_scale
        
        self.train_ds, self.test_ds = self.load_regression_data()

    
    def get_gp_curve_generator(self, testing: bool=False) -> Callable:
        """Returns a function that generates a batch of data for regression based on
        the original Conditional Neural Processes paper."""
        return partial(gen_from_arbitrary_gp,
                       batch_size=self.batch_size,
                       iterations=self.iterations,
                       min_kernel_length_scale=self.min_kernel_length_scale,
                       max_kernel_length_scale=self.max_kernel_length_scale,
                       min_num_context=self.min_num_context,
                       max_num_context=self.max_num_context,
                       min_num_target=self.min_num_target,
                       max_num_target=self.max_num_target,
                       min_x_val_uniform=self.min_x_val_uniform,
                       max_x_val_uniform=self.max_x_val_uniform,
                       testing=testing)


def draw_single_example_from_arbitrary_gp(
        kernel_length_scale: float,
        num_context: int,
        num_target: int,
        min_x_val_uniform: float=-2,
        max_x_val_uniform: float=2) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
    """Draw a single example from an arbitrary Gaussian Process with a null mean function and
    a ExponentiatedQuadratic kernel. The number of context and target points is deterministic.
    
    Note that the output is not batched.
    """    
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=kernel_length_scale)
    
    num_total_points = num_context + num_target
    
    x_values = tf.random.uniform(shape=(num_total_points, 1),
                                 minval=min_x_val_uniform,  # type: ignore
                                 maxval=max_x_val_uniform)
    
    gp = tfd.GaussianProcess(kernel, index_points=x_values, jitter=1.0e-4)
    y_values = tf.expand_dims(gp.sample(), axis=-1)
    
    idx = tf.random.shuffle(tf.range(num_total_points))
    
    # Select the targets which will consist of the context points
    # as well as some new target points
    target_x = x_values[:, :]
    target_y = y_values[:, :]

    # Select the observations
    context_x = tf.gather(x_values, indices=idx[:num_context], axis=0)
    context_y = tf.gather(y_values, indices=idx[:num_context], axis=0)
    
    return (context_x, context_y, target_x), target_y
