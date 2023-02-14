from typing import Callable, Dict, Tuple
from itertools import islice
from tqdm.auto import tqdm

import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

from utils.gaussian_processes.gp_hyperparameters import get_mean_fn, get_kernel

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def df_to_dataset(df_observed: pd.DataFrame, x_col: str, y_col: str) -> tf.data.Dataset:
    """Returns a tf.data.Dataset from a DataFrame with columns `x_col` and `y_col`."""
    ds_observed = tf.data.Dataset.from_tensor_slices(
        (df_observed[x_col].values.reshape(-1, 1), df_observed[y_col].values))
    return ds_observed


def get_gp_loss_fn(mean_fn: Callable,
                   kernel: tfp.math.psd_kernels.PositiveSemidefiniteKernel,
                   observation_noise_variance: float) -> Callable:
    """Returns a function that computes the negative log-likelihood of a Gaussian process."""
    
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
    """Plot the learning curve of the Gaussian process."""
    # TODO: Convert plot_function to matplotlib
    raise NotImplementedError("plot_learning_curve is not implemented yet.")


def train_gp(df_observed: pd.DataFrame,
             x_col: str,
             y_col: str,
             epochs: int,
             batch_size: int=128,
             plot: bool=False) -> Tuple[
                 Callable,
                 tfp.math.psd_kernels.PositiveSemidefiniteKernel,
                 Dict[str, tfp.util.TransformedVariable]]:
    """Train a Gaussian process on the observed data.
    
    Note that the batch_size is required as we update the GP's hyperparameters based
    on backpropagation through the loss function.
    """
    
    ds = df_to_dataset(df_observed, x_col=x_col, y_col=y_col)

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
    
    pbar = tqdm(enumerate(islice(batched_dataset, epochs)), total=epochs)
    
    for i, (index_points_batch, observations_batch) in pbar:
        # Run optimization for single batch
        with tf.GradientTape() as tape:
            loss = gp_loss_fn(index_points_batch, observations_batch)
        
        pbar.set_description(f"loss = {loss.numpy():.3f}")
        
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        batch_nlls.append((i, loss.numpy()))
        # Evaluate on all observations
        if i % 100 == 0:
            # Evaluate on all observed data
            ll = gp_loss_fn(
                index_points=df_observed[x_col].values.reshape(-1, 1),
                observations=df_observed[y_col].values)
            full_ll.append((i, ll.numpy()))
    
    if plot:
        plot_learning_curve(epochs, batch_nlls, full_ll)
    
    return mean_fn, kernel, variables


def variables_to_df(variables: Dict[str, tfp.util.TransformedVariable]) -> pd.DataFrame:
    """Format the variables as a DataFrame."""
    
    data = list([(name, var.numpy()) for name, var in variables.items()])
    df_variables = pd.DataFrame(
        data, columns=['Hyperparameters', 'Value'])
    return df_variables
