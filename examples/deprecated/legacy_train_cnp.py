#%%
import os
#os.chdir("c:Users/baker/Documents/MLMI4/conditional-neural-processes/")
import argparse
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp

from dataloader.load_regression_data_from_arbitrary_gp import RegressionDataGeneratorArbitraryGP
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb
from neural_process_model_conditional import NeuralProcessConditional
from utils.utility import PlotCallback

tfk = tf.keras
tfd = tfp.distributions


# Parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--epochs', type=int, default=120, help='Number of training epochs')
# parser.add_argument('-b', '--batch', type=int, default=1024, help='Batch size for training')
# parser.add_argument('-t', '--task', type=str, default='regression', help='Task to perform : (mnist|regression)')
# parser.add_argument('-c', '--num_context', type=int, default=100)
# parser.add_argument('-u', '--uniform_sampling', type=bool, default=True)
# args = parser.parse_args()

#args = argparse.Namespace(epochs=60, batch=1024, task="regression", num_context=100, uniform_sampling=True)
args = argparse.Namespace(epochs=10, batch=256, task="mnist", num_context=10, uniform_sampling=True)

# Note that num_context is not used for the 1D regression task.
#tf.config.set_visible_devices([], 'GPU')

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs


model_path = f'.data/CNP-5_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-{epoch:04d}.ckpt"


if args.task == 'mnist':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS  = load_mnist(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)
    
    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 2]



elif args.task == 'regression':
    data_generator = RegressionDataGeneratorArbitraryGP(
        iterations=100,
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



elif args.task == 'celeb':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_celeb(batch_size=BATCH_SIZE, num_context_points=args.num_context,
                                   uniform_sampling=args.uniform_sampling)

    # Model architecture
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 6]


# Compile model
model = NeuralProcessConditional(encoder_dims, decoder_dims)
model.compile(loss=NeuralProcessConditional.loss, optimizer='adam')


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

#%%

# Train
model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)


#%%