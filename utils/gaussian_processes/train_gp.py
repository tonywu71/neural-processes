from typing import Callable, Dict, Tuple
from itertools import islice
from tqdm.auto import tqdm

import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

from utils.gaussian_processes.init_gp import get_mean_fn, get_kernel

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def get_ds_observed(df_observed: pd.DataFrame, x_col: str, y_col: str) -> tf.data.Dataset:
    ds_observed = tf.data.Dataset.from_tensor_slices(
        (df_observed[x_col].values.reshape(-1, 1), df_observed[y_col].values))
    return ds_observed


def get_gp_loss_fn(mean_fn: Callable, kernel: tfp.math.psd_kernels.PositiveSemidefiniteKernel,
                   observation_noise_variance: float) -> Callable:
    
    # Use tf.function for more efficient function evaluation
    @tf.function(autograph=False, experimental_compile=False)
    def gp_loss_fn(index_points, observations):
        """Gaussian process negative-log-likelihood loss function."""
        gp = tfd.GaussianProcess(
            mean_fn=mean_fn,
            kernel=kernel,
            index_points=index_points,
            observation_noise_variance=observation_noise_variance
        )
        
        negative_log_likelihood = -gp.log_prob(observations)
        return negative_log_likelihood
    
    return gp_loss_fn


def plot_learning_curve(nb_iterations: int, batch_nlls, full_ll):
    # TODO: Convert plot_function to matplotlib
    pass


def train_gp(df_observed: pd.DataFrame,
             x_col: str,
             y_col: str,
             batch_size: int=128,
             plot: bool=False) -> Tuple[
                 Callable,
                 tfp.math.psd_kernels.PositiveSemidefiniteKernel,
                 Dict[str, tfp.util.TransformedVariable]]:
    ds = get_ds_observed(df_observed, x_col=x_col, y_col=y_col)

    mean_fn = get_mean_fn(df_observed, y_col=y_col)
    kernel, variables = get_kernel()
    
    observation_noise_variance = variables["observation_noise_variance"]
    trainable_variables = [v.variables[0] for v in variables.values()]
    
    batched_dataset = ds.shuffle(buffer_size=len(df_observed))\
                        .repeat(count=None)\
                        .batch(batch_size)
    
    gp_loss_fn = get_gp_loss_fn(mean_fn, kernel, observation_noise_variance)
    
    # Fit hyperparameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Training loop
    batch_nlls = []  # Batch NLL for plotting
    full_ll = []  # Full data NLL for plotting
    nb_iterations = 10001
    for i, (index_points_batch, observations_batch) in tqdm(
            enumerate(islice(batched_dataset, nb_iterations)), total=nb_iterations):
        # Run optimization for single batch
        with tf.GradientTape() as tape:
            loss = gp_loss_fn(index_points_batch, observations_batch)
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        batch_nlls.append((i, loss.numpy()))
        # Evaluate on all observations
        if i % 100 == 0:
            # Evaluate on all observed data
            ll = gp_loss_fn(
                index_points=df_observed.Date.values.reshape(-1, 1),
                observations=df_observed.CO2.values)
            full_ll.append((i, ll.numpy()))
    
    if plot:
        plot_learning_curve(nb_iterations, batch_nlls, full_ll)
    
    return mean_fn, kernel, variables


def variables_to_df(variables: Dict[str, tfp.util.TransformedVariable]) -> pd.DataFrame:
    data = list([(name, var.numpy()) for name, var in variables.items()])
    df_variables = pd.DataFrame(
        data, columns=['Hyperparameters', 'Value'])
    return df_variables
