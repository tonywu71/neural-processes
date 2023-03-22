from typing import List, Optional

import numpy as np
import pandas as pd

import tensorflow as tf

from dataloader.load_regression_data_from_arbitrary_gp_varying_kernel import draw_single_example_from_arbitrary_gp_varying_kernel

tfk = tf.keras

import tensorflow_probability as tfp
tfd = tfp.distributions

from neural_process_model_latent import NeuralProcessLatent
from nueral_process_model_conditional import NeuralProcessConditional

import matplotlib.pyplot as plt

from dataloader.load_regression_data_from_arbitrary_gp import draw_single_example_from_arbitrary_gp
from utils.gaussian_processes.plot_gp_utils import plot_mean_with_std, plot_preds_from_single_example


def plot_learning_curve(history: tf.keras.callbacks.History, filepath: Optional[str]=None):
    history_loss = pd.DataFrame(history.history, columns=["loss", "val_loss"])
    ax = history_loss.plot(xlabel="Epochs", ylabel="Validation loss (cross-entropy)",
                           title="Learning Curve")

    if filepath is not None:
        fig = ax.get_figure() # type: ignore
        fig.savefig(filepath)
    
    return


def plot_from_arbitrary_gp_wrt_context_points(model: tfk.Model,
                                              kernel_length_scale: float,
                                              list_num_context: List[int]) -> None:
    fig, axis = plt.subplots(len(list_num_context), 1,
                            figsize=(8, 3*len(list_num_context)),
                            sharex=True,
                            sharey=True)

    for idx_plot, num_context in enumerate(list_num_context):
        (context_x, context_y, target_x), target_y = draw_single_example_from_arbitrary_gp(
            kernel_length_scale=kernel_length_scale,
            num_context=num_context,
            num_target=50
        )
        
        plot_preds_from_single_example(model, context_x, context_y, target_x, target_y, ax=axis[idx_plot])

    return


def plot_from_arbitrary_gp_varying_kernel_wrt_context_points(model: tfk.Model,
                                                             kernel_length_scale: float,
                                                             list_num_context: List[int]) -> None:
    fig, axis = plt.subplots(len(list_num_context), 1,
                            figsize=(8, 3*len(list_num_context)),
                            sharex=True,
                            sharey=True)

    for idx_plot, num_context in enumerate(list_num_context):
        (context_x, context_y, target_x), target_y = draw_single_example_from_arbitrary_gp_varying_kernel(
            kernel_length_scale=kernel_length_scale,
            num_context=num_context,
            num_target=50
        )
        
        plot_preds_from_single_example(model, context_x, context_y, target_x, target_y, ax=axis[idx_plot])

    return


def plot_gp_vs_cnp_vs_lnp(cnp_model: NeuralProcessConditional,
                          lnp_model: NeuralProcessLatent,
                          kernel_length_scale: float,
                          list_num_context: List[int]):
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=kernel_length_scale)


    fig, axis = plt.subplots(3, len(list_num_context),
                            figsize=(3.5*len(list_num_context), 6),
                            sharex=True,
                            sharey=True)


    for idx_plot, num_context in enumerate(list_num_context):
        (context_x, context_y, target_x), target_y = draw_single_example_from_arbitrary_gp(
            kernel_length_scale=kernel_length_scale,
            num_context=num_context,
            num_target=100
        )
        
        # --- CNP ---
        plot_preds_from_single_example(cnp_model, context_x, context_y, target_x, target_y,
                                    show_title=False, ax=axis[1, idx_plot])  # type: ignore
        
        
        # --- LNP ---
        plot_preds_from_single_example(lnp_model, context_x, context_y, target_x, target_y,
                                    show_title=False, ax=axis[2, idx_plot])  # type: ignore
        
        # --- GP ---
        gp = tfd.GaussianProcessRegressionModel(
                    kernel=kernel,
                    index_points=target_x,
                    observation_index_points=context_x,
                    observations=tf.squeeze(context_y),
                    jitter=1.0e-4
        )

        context_x = tf.squeeze(context_x)
        target_x = tf.squeeze(target_x)
        gp_mean_predict = gp.mean()
        gp_std_predict = gp.stddev()
        
        idx_x_sorted = tf.argsort(target_x)
        
        target_x = tf.gather(target_x, idx_x_sorted)
        target_y = tf.gather(target_y, idx_x_sorted)
        gp_mean_predict = tf.gather(gp_mean_predict, idx_x_sorted)
        gp_std_predict = tf.gather(gp_std_predict, idx_x_sorted)

        plot_mean_with_std(x=target_x.numpy(),
                        mean=gp_mean_predict.numpy(),
                        std=gp_std_predict.numpy(),
                        y_true=target_y.numpy(),
                        ax=axis[0, idx_plot]  # type: ignore
        )
        
        axis[0, idx_plot].scatter(context_x.numpy(), context_y.numpy())  # type: ignore


    for ax, label in zip(axis[0, :], [f"n_context = {n}" for n in list_num_context]):  # type: ignore
        ax.set_title(label, size=20)

    for ax in axis.flatten()[:-1]:  # type: ignore
        ax.get_legend().remove()
        
    axis[0, 0].set_ylabel("GP", size=20)  # type: ignore
    axis[1, 0].set_ylabel("CNP", size=20)  # type: ignore
    axis[2, 0].set_ylabel("LNP", size=20)  # type: ignore

        
    fig.tight_layout()

    return


def plot_gp_vs_cnp_vs_lnp_varying_kernel(model: NeuralProcessConditional,
                                         lnp_model: NeuralProcessLatent,
                                         kernel_length_scale: float,
                                         list_num_context: List[int]):
    fig, axis = plt.subplots(3, len(list_num_context),
                            figsize=(3.5*len(list_num_context), 6),
                            sharex=True,
                            sharey=True)


    list_kernel_length_scale = []

    for idx_plot, num_context in enumerate(list_num_context):
        kernel_length_scale = np.random.uniform(0.2, 0.8)
        list_kernel_length_scale.append(kernel_length_scale)
        kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=kernel_length_scale)
        
        
        (context_x, context_y, target_x), target_y = draw_single_example_from_arbitrary_gp_varying_kernel(
            kernel_length_scale=kernel_length_scale,
            num_context=num_context,
            num_target=100
        )
        
        # --- CNP ---
        plot_preds_from_single_example(model, context_x, context_y, target_x, target_y,
                                    show_title=False, ax=axis[1, idx_plot])  # type: ignore
        
        
        # --- LNP ---
        plot_preds_from_single_example(lnp_model, context_x, context_y, target_x, target_y,
                                    show_title=False, ax=axis[2, idx_plot])  # type: ignore
        
        # --- GP ---
        gp = tfd.GaussianProcessRegressionModel(
                    kernel=kernel,
                    index_points=target_x,
                    observation_index_points=context_x,
                    observations=tf.squeeze(context_y),
                    jitter=1.0e-4
        )

        context_x = tf.squeeze(context_x)
        target_x = tf.squeeze(target_x)
        gp_mean_predict = gp.mean()
        gp_std_predict = gp.stddev()
        
        idx_x_sorted = tf.argsort(target_x)
        
        target_x = tf.gather(target_x, idx_x_sorted)
        target_y = tf.gather(target_y, idx_x_sorted)
        gp_mean_predict = tf.gather(gp_mean_predict, idx_x_sorted)
        gp_std_predict = tf.gather(gp_std_predict, idx_x_sorted)

        plot_mean_with_std(x=target_x.numpy(),
                        mean=gp_mean_predict.numpy(),
                        std=gp_std_predict.numpy(),
                        y_true=target_y.numpy(),
                        ax=axis[0, idx_plot]  # type: ignore
        )
        
        axis[0, idx_plot].scatter(context_x.numpy(), context_y.numpy())  # type: ignore



    list_labels = [f"n_context = {n}\nl = {l:.2f}" for n, l in zip(list_num_context, list_kernel_length_scale)]

    for ax, label in zip(axis[0, :], list_labels):  # type: ignore
        ax.set_title(label, size=18)

    for ax in axis.flatten()[:-1]:  # type: ignore
        ax.get_legend().remove()
        
    axis[0, 0].set_ylabel("GP", size=20)  # type: ignore
    axis[1, 0].set_ylabel("CNP", size=20)  # type: ignore
    axis[2, 0].set_ylabel("LNP", size=20)  # type: ignore

        
    fig.tight_layout()
    
    return
