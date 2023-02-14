#%%
import os
import argparse
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp

from dataloader.load_regression_data_uniform import RegressionDataGeneratorUniform
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

#tf.config.set_visible_devices([], 'GPU')

args = argparse.Namespace(epochs=15, batch=64, task='mnist', num_context=1, uniform_sampling=False)

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs


model_path = f'.data/model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-{epoch:04d}.ckpt"


if args.task == 'mnist':
    train_ds, test_ds = load_mnist(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)
    
    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)

elif args.task == 'regression':
    data_generator = RegressionDataGeneratorUniform()
    train_ds, test_ds = data_generator.load_regression_data(batch_size=BATCH_SIZE)

    # Model architecture
    encoder_dims = [128, 128, 128, 128]
    decoder_dims = [128, 128, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)

elif args.task == 'celeb':
    train_ds, test_ds = load_celeb(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)

    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 2]

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
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True,
                                                 verbose=1)
callbacks = [tensorboard_clbk, plot_clbk, cp_callback]



# Train
model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)


