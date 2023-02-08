from typing import Callable, Optional, Dict, Tuple
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from utils.gaussian_processes.train_gp import get_ds_observed, train_gp


class GPModel():
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
        self.df_observed = df_observed
        self.ds_observed = get_ds_observed(df_observed, x_col=x_col, y_col=y_col)
        self.mean_fn, self.kernel, self.variables = train_gp(df_observed, x_col, y_col,
                                                             epochs=epochs,
                                                             batch_size=self.batch_size,
                                                             plot=False)
        self.is_fitted = True

    
    def get_gp_posterior_predict(self, df_predict: pd.DataFrame, x_col: str) -> tfp.distributions.GaussianProcessRegressionModel:
        assert self.is_fitted, "Model is not fitted yet."
        assert set(df_predict.columns).issubset(self.df_observed.columns), "Column names do not match."
        
        # Posterior GP using fitted kernel and observed data
        gp_posterior_predict = tfd.GaussianProcessRegressionModel(
            mean_fn=self.mean_fn,
            kernel=self.kernel,
            index_points=df_predict[x_col].values.reshape(-1, 1),
            observation_index_points=self.df_observed[self.x_col].values.reshape(-1, 1),
            observations=self.df_observed[self.y_col].values,
            observation_noise_variance=self.variables["observation_noise_variance"])

        return gp_posterior_predict
    
    
    def sample_from_posterior_gp(self, df_predict: pd.DataFrame, x_col: str) -> tf.Tensor:
        """Note: gp_posterior_predict is recreated at each call.
        For efficiency, use get_gp_posterior_predict() to get the posterior GP and
        then sample manually without this method."""
        gp_posterior_predict = self.get_gp_posterior_predict(df_predict, x_col)
        return gp_posterior_predict.sample()
    
    
    @staticmethod
    def plot_from_predictions(gp_posterior_predict: tfp.distributions.GaussianProcessRegressionModel):
        # Posterior mean and standard deviation
        posterior_mean_predict = gp_posterior_predict.mean()
        posterior_std_predict = gp_posterior_predict.stddev()
        
        # TODO: plot
        pass
