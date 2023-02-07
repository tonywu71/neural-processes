from typing import Tuple, Callable, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def get_mean_fn(df_observed: pd.DataFrame, y_col: str) -> Callable:
    observations_mean = tf.constant(
        [np.mean(df_observed[y_col].values)], dtype=tf.float64)
    mean_fn = (lambda _: observations_mean)
    return mean_fn


def get_kernel() -> Tuple[tfp.math.psd_kernels.PositiveSemidefiniteKernel, Dict[str, tfp.util.TransformedVariable]]:
    # Define the kernel with trainable parameters. 
    # Note we transform some of the trainable variables to ensure
    #  they stay positive.

    # Use float64 because this means that the kernel matrix will have 
    #  less numerical issues when computing the Cholesky decomposition

    # Constrain to make sure certain parameters are strictly positive
    constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    # Smooth kernel hyperparameters
    smooth_amplitude = tfp.util.TransformedVariable(
        initial_value=10., bijector=constrain_positive, dtype=np.float64,
        name='smooth_amplitude')
    smooth_length_scale = tfp.util.TransformedVariable(
        initial_value=10., bijector=constrain_positive, dtype=np.float64,
        name='smooth_length_scale')
    # Smooth kernel
    smooth_kernel = tfk.ExponentiatedQuadratic(
        amplitude=smooth_amplitude, 
        length_scale=smooth_length_scale)

    # Local periodic kernel hyperparameters
    periodic_amplitude = tfp.util.TransformedVariable(
        initial_value=5.0, bijector=constrain_positive, dtype=np.float64,
        name='periodic_amplitude')
    periodic_length_scale = tfp.util.TransformedVariable(
        initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
        name='periodic_length_scale')
    periodic_period = tfp.util.TransformedVariable(
        initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
        name='periodic_period')
    periodic_local_length_scale = tfp.util.TransformedVariable(
        initial_value=1.0, bijector=constrain_positive, dtype=np.float64,
        name='periodic_local_length_scale')
    # Local periodic kernel
    local_periodic_kernel = (
        tfk.ExpSinSquared(
            amplitude=periodic_amplitude, 
            length_scale=periodic_length_scale,
            period=periodic_period) * 
        tfk.ExponentiatedQuadratic(
            length_scale=periodic_local_length_scale))

    # Short-medium term irregularities kernel hyperparameters
    irregular_amplitude = tfp.util.TransformedVariable(
        initial_value=1., bijector=constrain_positive, dtype=np.float64,
        name='irregular_amplitude')
    irregular_length_scale = tfp.util.TransformedVariable(
        initial_value=1., bijector=constrain_positive, dtype=np.float64,
        name='irregular_length_scale')
    irregular_scale_mixture = tfp.util.TransformedVariable(
        initial_value=1., bijector=constrain_positive, dtype=np.float64,
        name='irregular_scale_mixture')
    # Short-medium term irregularities kernel
    irregular_kernel = tfk.RationalQuadratic(
        amplitude=irregular_amplitude,
        length_scale=irregular_length_scale,
        scale_mixture_rate=irregular_scale_mixture)

    # Noise variance of observations
    # Start out with a medium-to high noise
    observation_noise_variance = tfp.util.TransformedVariable(
        initial_value=1, bijector=constrain_positive, dtype=np.float64,
        name='observation_noise_variance')

    variables = {
        "smooth_amplitude": smooth_amplitude,
        "smooth_length_scale": smooth_length_scale,
        "periodic_amplitude": periodic_amplitude,
        "periodic_length_scale": periodic_length_scale,
        "periodic_period": periodic_period,
        "periodic_local_length_scale": periodic_local_length_scale,
        "irregular_amplitude": irregular_amplitude,
        "irregular_length_scale": irregular_length_scale,
        "irregular_scale_mixture": irregular_scale_mixture,
        "observation_noise_variance": observation_noise_variance
    }

    # Sum all kernels to single kernel containing all characteristics
    kernel = (smooth_kernel + local_periodic_kernel + irregular_kernel)
    
    return kernel, variables
