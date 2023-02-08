import os
import argparse
from datetime import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from model import ConditionalNeuralProcess
from utils import PlotCallback, get_gp_curve_generator

tfk = tf.keras
tfd = tfp.distributions

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=15, help='Number of training epochs')
parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size for training')
parser.add_argument('-t', '--task', type=str, default='mnist', help='Task to perform : (mnist|regression)')

args = parser.parse_args()

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs

if args.task == 'mnist':
    # Dataset
    mnist = tfds.load('mnist')
    train_ds, test_ds = mnist['train'], mnist['test']

    def encode(element):
        # element should be already batched
        img = tf.cast(element['image'], tf.float32) / 255. # normalise pixels within [0,1] range
        batch_size = tf.shape(img)[0] 
        
        # Number observations of the target image randomly chosen between [10,100]
        num_context = tf.random.uniform(
            shape=[], minval=10, maxval=100, dtype=tf.int32)
        
        # For each of our observations, sample x,y coordinates in range [0,27]
        context_x = tf.random.uniform(
            shape=(batch_size, num_context, 2),
            minval=0, maxval=27, dtype=tf.int32)
        
        # Sample observation coordinates from target image
        context_y = tf.gather_nd(img, context_x, batch_dims=1)

        # Normalise the observation coordinates to the range [0,27] to be model size agnostic
        context_x = tf.cast(context_x, tf.float32)  /27.

        # define the grid of x,y coordinates for every pixel
        cols, rows = tf.meshgrid(tf.range(28.), tf.transpose(tf.range(28.)))

        # combine the x,y coordinate arrays into a single array
        grid = tf.stack([rows, cols], axis=-1)  # (28, 28, 2)

        # copy observation coordinates across the entire batch
        batch_grid = tf.tile(
            tf.expand_dims(grid, axis=0),
            [batch_size, 1, 1, 1])  # (batch_size, 28, 28, 2)
        
        # Normalise the observation coordinates to the range [0,27] to be model size agnostic
        target_x = tf.reshape(
            batch_grid, (batch_size, 28 * 28, 2)) / 27.  # normalize
        
        # reshape the target image to have shape: batch size, input dim, 1 
        target_y = tf.reshape(img, (batch_size, 28 * 28, 1))
        return (context_x, context_y, target_x), target_y

    train_ds = train_ds.batch(BATCH_SIZE).map(encode)
    test_ds = test_ds.batch(1).map(encode)

    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)

else: # args.task == regression
    train_ds = tf.data.Dataset.from_generator(
        get_gp_curve_generator(
            iterations=250,
            batch_size=BATCH_SIZE,
            max_num_context=10,
            testing=False),
        output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
    )
    test_ds = tf.data.Dataset.from_generator(
        get_gp_curve_generator(
            iterations=250,
            batch_size=1,
            testing=True),
        output_types=((tf.float32, tf.float32, tf.float32), tf.float32)
    )

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

# Train
model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)
