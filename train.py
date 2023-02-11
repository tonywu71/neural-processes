#%%
import os
import argparse
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp

from dataloader.load_regression_data_from_arbitrary_gp import RegressionDataGeneratorArbitraryGP
from dataloader.load_mnist import load_mnist
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

args = argparse.Namespace(epochs=15, batch=64, task='mnist')

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs


if args.task == 'mnist':
    train_ds, test_ds = load_mnist(batch_size=BATCH_SIZE)
    
    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)

else: # args.task == regression
    data_generator = RegressionDataGeneratorArbitraryGP()
    train_ds, test_ds = data_generator.load_regression_data(batch_size=BATCH_SIZE)

    # Model architecture
    encoder_dims = [128, 128, 128, 128]
    decoder_dims = [128, 128, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)

# Compile model
model = ConditionalNeuralProcess(encoder_dims, decoder_dims)
model.compile(loss=loss, optimizer='adam')

# Callbacks
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'cnp', args.task, time)
tensorboard_clbk = tfk.callbacks.TensorBoard(
    log_dir=log_dir, update_freq='batch')
plot_clbk = PlotCallback(logdir=log_dir, ds=test_ds, task=args.task)
callbacks = [tensorboard_clbk, plot_clbk]

#%%

# Train
model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)

#%%
