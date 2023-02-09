from typing import Callable, Optional, Dict

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

import matplotlib.pyplot as plt

from utils.gaussian_processes.train_gp import df_to_dataset, train_gp


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

    
    def get_gp_posterior_predict(self, df_predict: pd.DataFrame) -> tfp.distributions.GaussianProcessRegressionModel:
        """Returns the posterior GP for the given data as a tfp.distributions.GaussianProcessRegressionModel object."""
        assert self.is_fitted, "Model is not fitted yet."
        assert set(df_predict.columns).issubset(self.df_observed.columns), "Column names do not match."
        
        # Posterior GP using fitted kernel and observed data
        gp_posterior_predict = tfd.GaussianProcessRegressionModel(
            mean_fn=self.mean_fn,
            kernel=self.kernel,
            index_points=df_predict[self.x_col].values.reshape(-1, 1),
            observation_index_points=self.df_observed[self.x_col].values.reshape(-1, 1),
            observations=self.df_observed[self.y_col].values,
            observation_noise_variance=self.variables["observation_noise_variance"])

        return gp_posterior_predict
    
    
    def sample_from_posterior_gp(self, df_predict: pd.DataFrame) -> tf.Tensor:
        """Samples from the posterior GP for the given data. Note that the x_col
        should match the one used for fitting the model.
        
        NB: gp_posterior_predict is recreated at each call.
        For efficiency, use get_gp_posterior_predict() to get the posterior GP and
        then sample manually without this method."""
        gp_posterior_predict = self.get_gp_posterior_predict(df_predict)
        return gp_posterior_predict.sample()
    
    
    @staticmethod
    def plot_from_predictions(gp_posterior_predict: tfp.distributions.GaussianProcessRegressionModel):
        # Posterior mean and standard deviation
        posterior_mean_predict = gp_posterior_predict.mean()
        posterior_std_predict = gp_posterior_predict.stddev()
        
        # Plot posterior mean and standard deviation
        
        
        # TODO: plot
        raise NotImplementedError("plot_from_predictions() not implemented yet.")


def plot_mean_with_std(x: np.ndarray,
                       mean: np.ndarray,
                       std: np.ndarray,
                       ax: Optional[plt.Axes]=None,
                       label: Optional[str]=None,
                       alpha: float=0.3) -> plt.Axes:
    """Plot mean and standard deviation."""
    if ax is None:
        ax = plt.gca()
    
    # Plot mean
    ax.plot(x, mean, label=label)
    
    # Plot standard deviation
    ax.fill_between(x,
                    mean - std,
                    mean + std,
                    alpha=alpha)
    
    return ax
