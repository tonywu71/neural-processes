from typing import Callable, Optional, Dict

import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import matplotlib.pyplot as plt

from utils.gaussian_processes.train_gp import df_to_dataset, train_gp
from utils.gaussian_processes.plot_gp_utils import plot_mean_with_std


class GPModel():
    """Class for Gaussian process regression."""
    def __init__(self, batch_size: int=128) -> None:
        self.batch_size = batch_size
        
        self.is_fitted = False
        self.df_observed: Optional[pd.DataFrame] = None
        self.ds_observed: Optional[tf.data.Dataset] = None
        self.x_col: Optional[str] = None
        self.y_col: Optional[str] = None
        self.mean_fn: Optional[Callable] = None
        self.kernel: Optional[tfp.math.psd_kernels.PositiveSemidefiniteKernel] = None
        self.variables: Optional[Dict[str, tfp.util.TransformedVariable]] = None
    
    
    def fit(self, df_observed: pd.DataFrame, x_col: str, y_col: str, epochs: int):
        """Fit the Gaussian process model to the observed data."""
        self.df_observed = df_observed
        self.x_col = x_col
        self.y_col = y_col
        self.ds_observed = df_to_dataset(df_observed, x_col=self.x_col, y_col=self.y_col)
        self.mean_fn, self.kernel, self.variables = train_gp(df_observed, self.x_col, self.y_col,
                                                             epochs=epochs,
                                                             batch_size=self.batch_size,
                                                             plot=False)
        self.is_fitted = True

    
    def get_gp_posterior_predict(self) -> tfp.distributions.GaussianProcessRegressionModel:
        """Returns the posterior GP for the given data as a tfp.distributions.GaussianProcessRegressionModel object."""
        assert self.is_fitted, "Model is not fitted yet."
        
        # Posterior GP using fitted kernel and observed data
        index_points = self.df_observed[self.x_col].values.reshape(-1, 1)  # type: ignore
        observations = self.df_observed[self.y_col].values  # type: ignore
        
        gp_posterior_predict = tfd.GaussianProcessRegressionModel(
            mean_fn=self.mean_fn,
            kernel=self.kernel,
            index_points=index_points,
            observation_index_points=index_points,
            observations=observations,
            observation_noise_variance=self.variables["observation_noise_variance"])  # type: ignore

        return gp_posterior_predict
    
    
    def plot_posterior(self) -> plt.Axes:
        gp_posterior_predict = self.get_gp_posterior_predict()
        x = tf.squeeze(gp_posterior_predict.index_points)
        ax = plot_mean_with_std(x=x.numpy(),
                                mean=gp_posterior_predict.mean().numpy(),
                                std=gp_posterior_predict.stddev().numpy())

        self.df_observed.plot(x=self.x_col, y=self.y_col, ax=ax, label="Original")  # type: ignore
        ax.legend()
        return ax

    
    def sample(self, x_values: tf.Tensor) -> tf.Tensor:
        """Draws samples from a Gaussian Process defined by a mean function and a covariance function."""
        gp = tfd.GaussianProcess(
            index_points=x_values,
            mean_fn=self.mean_fn,
            kernel=self.kernel)
        
        return gp.sample()
    