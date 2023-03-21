#%%
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp

import sys
sys.path.append('example-cnp/')

from dataloader.load_regression_data_from_arbitrary_gp import RegressionDataGeneratorArbitraryGP
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb

from neural_process_model_hybrid import NeuralProcessHybrid
from neural_process_model_latent import NeuralProcessLatent
#from nueral_process_model_conditional import ConditionalNeuralProcess as NeuralProcess
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

#args = argparse.Namespace(epochs=60, batch=1024, task='regression', num_context=1000, uniform_sampling=True, model='LNP')
args = argparse.Namespace(epochs=30, batch=256, task='mnist', num_context=100, uniform_sampling=True, model='LNP')

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs
LOG_PRIORS = True

model_path = f'.data/{args.model}_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-{epoch:04d}.ckpt"

TRAINING_ITERATIONS = int(100) # 1e5
TEST_ITERATIONS = int(TRAINING_ITERATIONS/5)
if args.task == 'mnist':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_mnist(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)
    
    # Model architecture
    z_output_sizes = [500, 500, 500, 1000]
    enc_output_sizes = [500, 500, 500, 500]
    dec_output_sizes = [500, 500, 500, 2]


elif args.task == 'regression':
    data_generator = RegressionDataGeneratorArbitraryGP(
        iterations=TRAINING_ITERATIONS,
        n_iterations_test=TEST_ITERATIONS,
        batch_size=BATCH_SIZE,
        min_num_context=3,
        max_num_context=40,
        min_num_target=2,
        max_num_target=40,
        min_x_val_uniform=-2,
        max_x_val_uniform=2,
        kernel_length_scale=0.4,
        
    )
    train_ds, test_ds = data_generator.load_regression_data()

    # Model architecture
    z_output_sizes = [500, 500, 500, 1000]
    enc_output_sizes = [500, 500, 500, 500]
    dec_output_sizes = [500, 500, 500, 2]    


elif args.task == 'celeb':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_celeb(batch_size=BATCH_SIZE, num_context_points=args.num_context,
                                   uniform_sampling=args.uniform_sampling)

    # Model architecture
    z_output_sizes = [500, 500, 500, 1000]
    enc_output_sizes = [500, 500, 500, 500]
    dec_output_sizes = [500, 500, 500, 6]







# Define NP Model
if args.model == 'LNP':
    model = NeuralProcessLatent(z_output_sizes, enc_output_sizes, dec_output_sizes)
elif args.model == 'HNP':
    model = NeuralProcessHybrid(z_output_sizes, enc_output_sizes, dec_output_sizes)

optimizer = tf.keras.optimizers.Adam(1e-3)

# Callbacks
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'np', args.model, args.task, time)
writer = tf.summary.create_file_writer(log_dir)
plot_clbk = PlotCallback(logdir=log_dir, ds=test_ds, task=args.task)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True)
callbacks = [plot_clbk, cp_callback]
for callback in callbacks: callback.model = model





def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    


    with tf.GradientTape() as tape:
        loss = model.compute_loss(x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return tf.math.reduce_mean(loss)




#%%
epochs = args.epochs
epochs=100

for epoch in range(31, epochs + 1):
    with tqdm(total=TRAINING_ITERATIONS, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        train_loss = tf.keras.metrics.Mean()
        for idx, train_x in enumerate(train_ds):

            # if idx == 5:
            #     tf.profiler.experimental.start(log_dir)
            #     print("profiling!")
            # tf.summary.trace_on(graph=True) # Uncomment to trace the computational graph

            loss = train_step(model, train_x, optimizer)

            # if idx == 5:
            #     tf.profiler.experimental.stop(log_dir)
            #     print("profiling end!")

            train_loss(loss)
            tepoch.set_postfix({'Batch': idx, 'Train Loss': train_loss.result().numpy()})
            tepoch.update(1)
            with writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=epoch*TRAINING_ITERATIONS + idx)
                # tf.summary.trace_export(
                #     name=f"NP_trace {idx}",
                #     step=idx) # Uncomment to trace the computational graph

            


        test_loss = tf.keras.metrics.Mean()
        for idx, test_x in enumerate(test_ds):
            loss = model.compute_loss(test_x)

            test_loss(loss)
            tepoch.set_postfix({'Batch': idx, 'Test Loss': test_loss.result().numpy()})
            with writer.as_default():
                tf.summary.scalar('test_loss', test_loss.result(), step=epoch*TEST_ITERATIONS + idx)

        if LOG_PRIORS:
            (context_x, context_y, target_x), target_y = next(iter(test_ds))
            context = tf.concat((context_x, context_y), axis=-1)
            hidden = model.z_encoder_latent.model(context)

            mu, log_sigma = tf.split(hidden, num_or_size_splits=2, axis=-1) # split the output in half
            sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

            with writer.as_default():
                tf.summary.histogram('prior/mu', mu, epoch)
                tf.summary.histogram('prior/sigma', sigma, epoch)

            target_context = tf.concat((target_x, target_y), axis=2)
            hidden = model.z_encoder_latent.model(target_context)
            mu, log_sigma = tf.split(hidden, num_or_size_splits=2, axis=-1) # split the output in half
            sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

            with writer.as_default():
                tf.summary.histogram('posterior/mu', mu, epoch)
                tf.summary.histogram('posterior/sigma', sigma, epoch)
            print("logged priors")



        tepoch.set_postfix({'Train loss': train_loss.result().numpy(), 'Test loss': test_loss.result().numpy()})
        
        for callback in callbacks: callback.on_epoch_end(epoch, logs={'loss': train_loss.result()})
        writer.flush()

#%%


