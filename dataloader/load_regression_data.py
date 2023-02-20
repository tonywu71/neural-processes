from functools import partial
from typing import Optional, Tuple, Callable, Iterator

import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from dataloader.regression_data_generator_base import RegressionDataGeneratorBase
from utils.gaussian_processes.gp_model import GPModel


DEFAULT_TESTING_NUM_TARGET = 400


def gen_from_gp(
        gp_model: GPModel,
        batch_size,
        iterations,
        min_num_context,
        max_num_context,
        min_num_target,
        max_num_target,
        min_x_val_uniform,
        max_x_val_uniform,
        testing) -> Iterator[Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]]:     
    """Generates a batch of data for regression based on an instance of GPModel
    (i.e. a fitted Gaussian Process)."""     
    
    for _ in range(iterations):
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
            num_target = DEFAULT_TESTING_NUM_TARGET
        
        num_total_points = num_context + num_target
        
        x_values = tf.random.uniform(shape=(batch_size, num_total_points, 1),
                                     minval=min_x_val_uniform,
                                     maxval=max_x_val_uniform)
        
        y_values = tf.expand_dims(gp_model.sample(x_values=x_values), axis=-1)
        
        idx = tf.random.shuffle(tf.range(num_total_points))
        
        # Select the targets which will consist of the context points
        # as well as some new target points
        target_x = x_values[:, :, :]
        target_y = y_values[:, :, :]

        # Select the observations
        context_x = tf.gather(x_values, indices=idx[:num_context], axis=1)
        context_y = tf.gather(y_values, indices=idx[:num_context], axis=1)

        yield (context_x, context_y, target_x), target_y


class RegressionDataGenerator(RegressionDataGeneratorBase):
    """Class that generates regression data from a Gaussian Process defined by a GPModel instance."""
    def __init__(self,
                 gp_model: GPModel,
                 df_predict: pd.DataFrame,
                 iterations: int=250,
                 batch_size: int=32,
                 min_num_context: int=3,
                 max_num_context: int=10,
                 min_num_target: int=2,
                 max_num_target: int=10,
                 min_x_val_uniform: Optional[int]=None,
                 max_x_val_uniform: Optional[int]=None,
                 n_iterations_test: Optional[int]=None):

        self.gp_model = gp_model
        assert self.gp_model.is_fitted, "GP model must be fitted before using it to generate data."
        
        self.x_col = gp_model.x_col
        self.y_col = gp_model.y_col
        self.df_predict = df_predict
        
        assert self.x_col in self.df_predict.columns, f"df_predict must contain column {self.x_col}"
        assert self.y_col in self.df_predict.columns, f"df_predict must contain column {self.y_col}"
        
        # If min_x_val_uniform and max_x_val_uniform are not specified, use the min and max values
        # from the training data (cf df_observed):
        if min_x_val_uniform is None:
            min_x_val_uniform = self.df_predict[self.x_col].min()  # type: ignore
        if max_x_val_uniform is None:
            max_x_val_uniform = self.df_predict[self.x_col].max()  # type: ignore
        
        super().__init__(iterations=iterations,
                         batch_size=batch_size,
                         min_num_context=min_num_context,
                         max_num_context=max_num_context,
                         min_num_target=min_num_target,
                         max_num_target=max_num_target,
                         min_x_val_uniform=min_x_val_uniform,  # type: ignore
                         max_x_val_uniform=max_x_val_uniform,  # type: ignore
                         n_iterations_test=n_iterations_test)
        
        self.train_ds, self.test_ds = self.load_regression_data()
        self.test_ds = self.test_ds.take(self.n_iterations_test)
        

    def get_gp_curve_generator(self, testing: bool=False) -> Callable:
        """Returns a generator function that generates regression data from a Gaussian Process."""
        return partial(gen_from_gp,
                       gp_model=self.gp_model,
                       batch_size=self.batch_size,
                       iterations=self.iterations,
                       min_num_context=self.min_num_context,
                       max_num_context=self.max_num_context,
                       min_num_target=self.min_num_target,
                       max_num_target=self.max_num_target,
                       min_x_val_uniform=self.min_x_val_uniform,
                       max_x_val_uniform=self.max_x_val_uniform,
                       testing=testing)
