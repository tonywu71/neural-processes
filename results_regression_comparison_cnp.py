#%%
import os
os.chdir("/Users/baker/Documents/MLMI4/conditional-neural-processes/")
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from dataloader.load_regression_data_from_arbitrary_gp import RegressionDataGeneratorArbitraryGP
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb
from model import ConditionalNeuralProcess
from utility import PlotCallback

tfk = tf.keras
tfd = tfp.distributions


# # Parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--epochs', type=int, default=15, help='Number of training epochs')
# parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size for training')
# parser.add_argument('-t', '--task', type=str, default='mnist', help='Task to perform : (mnist|regression)')

# args = parser.parse_args()

args = argparse.Namespace(epochs=120, batch=1024, task="regression", num_context=10, uniform_sampling=True)
# Note that num_context is not used for the 1D regression task.
tf.config.set_visible_devices([], 'GPU') # DONT use the GPU, not needed

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs



if args.task == 'mnist':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS  = load_mnist(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)
    
    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)


elif args.task == 'regression':
    data_generator = RegressionDataGeneratorArbitraryGP(
        iterations=25,
        batch_size=BATCH_SIZE,
        min_num_context=3,
        max_num_context=40,
        min_num_target=2,
        max_num_target=40,
        min_x_val_uniform=-2,
        max_x_val_uniform=2,
        kernel_length_scale=0.4
    )
    train_ds, test_ds = data_generator.load_regression_data()

    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)


elif args.task == 'celeb':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_celeb(batch_size=BATCH_SIZE, num_context_points=args.num_context,
                                   uniform_sampling=args.uniform_sampling)

    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 6]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)


# Compile model
model = ConditionalNeuralProcess(encoder_dims, decoder_dims)


model_path = ".data/CNP2_model_regression_context_10_uniform_sampling_True/cp-0075.ckpt"
model.load_weights(model_path)

#%%

def plot_mean_with_std(x: np.ndarray,
                       mean: np.ndarray,
                       std: np.ndarray,
                       y_true=None,
                       ax=None,
                       alpha: float=0.3) -> plt.Axes:
    """Plot mean and standard deviation.
    IMPORTANT: x should be sorted and mean and std should be sorted in the same order as x."""
    
    if ax is None:
        ax = plt.gca()
    
    # Plot mean:
    ax.plot(x, mean, label="Mean prediction")  # type: ignore
    
    # Plot target if given:
    if y_true is not None:
        ax.plot(x, y_true, label="True function")
    
    # Plot standard deviation:
    ax.fill_between(x,  # type: ignore
                    mean - std,  # type: ignore
                    mean + std,  # type: ignore
                    alpha=alpha,
                    label="Standard deviation")
    
    ax.legend()
    
    return ax  # type: ignore
def plot_regression(target_x, target_y, context_x, context_y, pred_y):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    
    x_val = tf.squeeze(target_x[0, :, :])
    pred_y = tf.squeeze(pred_y[0, :, :])
    y_true = tf.squeeze(target_y[0, :, :])
    
    idx_x_sorted = tf.argsort(x_val)
    
    x_val = tf.gather(x_val, idx_x_sorted)
    mean = tf.gather(pred_y[:, 0], idx_x_sorted)
    std = tf.gather(pred_y[:, 1], idx_x_sorted)
    y_true = tf.gather(y_true, idx_x_sorted)
    
    plot_mean_with_std(x=x_val, mean=mean, std=std, y_true=y_true, ax=ax)
    ax.scatter(context_x[0, :, :], context_y[0, :, :])  # type: ignore
    
    return fig


# model_path = f'.data/{args.model}_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' \
#                     + "cp-.ckpt"


            
tf.random.set_seed(3)
test_iter = iter(test_ds)
x = next(test_iter)

(context_x, context_y, target_x), target_y = x
pred_y = model(x[0])
fig = plot_regression(target_x, target_y, context_x, context_y, pred_y)
#fig.suptitle(f'loss {logs["loss"]:.5f}')

#%%