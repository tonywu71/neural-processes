from functools import partial
from typing import Tuple, Callable, Iterator

import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from dataloader.regression_data_generator_base import RegressionDataGeneratorBase
from utils.gaussian_processes.gp_model import GPModel


def gen_from_gp(
        gp_model: GPModel,
        df_predict: pd.DataFrame,
        batch_size,
        iterations,
        min_num_context,
        max_num_context,
        min_num_target) -> Iterator[Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]]:     
    """Generates a batch of data for regression based on an instance of GPModel
    (i.e. a fitted Gaussian Process)."""     
    
    gp_posterior_predict = gp_model.get_gp_posterior_predict(df_predict)
    
    for _ in range(iterations):
        # NB: The distribution of y_values is the same for each iteration (i.e. the the one defined by
        #     the arbitrary GP) but the sampled x_values do differ (in terms of size and values).
        num_total_points = len(gp_posterior_predict.index_points)
        
        num_context = tf.random.uniform(shape=[],
                            minval=min_num_context,
                            maxval=max_num_context,
                            dtype=tf.int32)
        
        num_target = tf.random.uniform(shape=[],
                                    minval=min_num_target,
                                    maxval=num_total_points-num_context,
                                    dtype=tf.int32)
        
        x_values = tf.tile(tf.reshape(gp_posterior_predict.index_points,
                                      shape=(1, num_total_points, -1)),
                           multiples=(batch_size, 1, 1))
        y_values = tf.expand_dims(gp_posterior_predict.sample(sample_shape=(batch_size,)), axis=-1)
        
        # Select the targets which will consist of the context points
        # as well as some new target points
        idx = tf.random.shuffle(tf.range(num_total_points))
        
        # Select the observations (randomly select num_context examples from x_values and y_values)
        context_x = tf.gather(x_values, idx[:num_context], axis=1)
        context_y = tf.gather(y_values, idx[:num_context], axis=1)
        
        target_x = tf.gather(x_values, idx[:num_target+num_context], axis=1)
        target_y = tf.gather(y_values, idx[:num_target+num_context], axis=1)
        
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
                 min_num_target: int=2):
        super().__init__(iterations=iterations, batch_size=batch_size)
        
        self.gp_model = gp_model
        assert self.gp_model.is_fitted, "GP model must be fitted before using it to generate data."
        
        self.x_col = gp_model.x_col
        self.y_col = gp_model.y_col
        self.df_predict = df_predict
        
        assert self.x_col in self.df_predict.columns, f"df_predict must contain column {self.x_col}"
        assert self.y_col in self.df_predict.columns, f"df_predict must contain column {self.y_col}"
        
        self.num_total_points = len(self.df_predict)
        
        self.min_num_context = min_num_context
        self.max_num_context = max_num_context
        self.min_num_target = min_num_target
        
        assert min_num_context <= max_num_context <= self.num_total_points, f"min_num_context={min_num_context} must be <= max_num_context={max_num_context} <= num_total_points={self.num_total_points}"
        assert self.min_num_target <= self.num_total_points, f"min_num_target={self.min_num_target} must be <= num_total_points={self.num_total_points}"
        
        self.train_ds, self.test_ds = self.load_regression_data()


    def get_gp_curve_generator(self) -> Callable:
        """Returns a generator function that generates regression data from a Gaussian Process."""
        return partial(gen_from_gp,
                       gp_model=self.gp_model,
                       df_predict=self.df_predict,
                       batch_size=self.batch_size,
                       iterations=self.iterations,
                       min_num_context=self.min_num_context,
                       max_num_context=self.max_num_context,
                       min_num_target=self.min_num_target)
    
    
    def load_regression_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Loads regression data from a Gaussian Process."""
        train_ds = tf.data.Dataset.from_generator(
            self.get_gp_curve_generator(),
            output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
        )
        test_ds = tf.data.Dataset.from_generator(
            self.get_gp_curve_generator(),
            output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
        )
        
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)  # No need to shuffle as the data is already generated randomly
        test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        return train_ds, test_ds
