#%%
import os
os.chdir("/Users/baker/Documents/MLMI4/conditional-neural-processes/")
import argparse
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_probability as tfp

import sys
sys.path.append('example-cnp/')

from dataloader.load_regression_data_from_arbitrary_gp import RegressionDataGeneratorArbitraryGP
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb

from neural_process_model_hybrid import NeuralProcessHybrid
from neural_process_model_latent import NeuralProcessLatent
from utils.utility import PlotCallback

import matplotlib.pyplot as plt

tfk = tf.keras
tfd = tfp.distributions





args = argparse.Namespace(epochs=60, batch=256, task='mnist', num_context=100, uniform_sampling=True, model='LNP')


BATCH_SIZE = args.batch
EPOCHS = args.epochs


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


#%%
import numpy as np

fig, axs = plt.subplots(3, 4, figsize=(10, 5))
#for i, num_context in enumerate([1,10,100,1000]):#([1,10,100,1000]):
for i, num_context in enumerate([1,10,100,1000]):#([1,10,100,1000]):

    #model.load_weights(f'trained_models/model_{args.task}_context_{num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0015.ckpt")
    #model.load_weights(f'.data/CNP2_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0010.ckpt")
    #model.load_weights(f'.data/CNP2_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0010.ckpt")
    

    if args.task == 'celeb':
        model.load_weights(f'.data/CNP2_model_celeb_context_{num_context}_uniform_sampling_True/cp-0010.ckpt')
        train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_celeb(batch_size=BATCH_SIZE, num_context_points=num_context, uniform_sampling=args.uniform_sampling)
        img_size=32

        it = iter(test_ds)
        next(it)
        next(it)
        next(it)
        (context_x, context_y, target_x), target_y = next(it)
        pred_y = model((context_x, context_y, target_x))

        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        # Plot context points
        white_img = tf.tile(tf.constant([[[0.,0.,0.]]]), [img_size, img_size, 1])
        indices = tf.cast(context_x[0] * float(img_size - 1.0), tf.int32)

        updates = context_y[0]

        context_img = tf.tensor_scatter_nd_update(white_img, indices, updates)
        axs[0][i].imshow(context_img.numpy())
        axs[0][i].axis('off')
        axs[0][i].set_title(f'{num_context} context points')
        # Plot mean and variance
        mean = tf.reshape(mu[0], (img_size, img_size, 3))
        var = tf.reshape(sigma[0], (img_size, img_size, 3))

        axs[1][i].imshow(mean.numpy(), vmin=0., vmax=1.)
        axs[2][i].imshow(var.numpy(), vmin=0., vmax=1.)
        axs[1][i].axis('off')
        axs[2][i].axis('off')
        axs[1][i].set_title('Predicted mean')
        axs[2][i].set_title('Predicted variance')

    elif args.task == 'mnist':
        # epochs = os.listdir(model_path[:len("cp-{epoch:04d}.ckpt")*-1])
        # epochs = max([m[3:7] for m in epochs if '.data' in m])
        # model.load_weights(model_path.format(epoch=epochs))
        #model.load_weights('.data/LNP_model_mnist_context_10_uniform_sampling_True/cp-0027.ckpt')
        model.load_weights('.data/LNP_model_mnist_context_100_uniform_sampling_True/cp-0092.ckpt')
        train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_mnist(batch_size=BATCH_SIZE, num_context_points=num_context, uniform_sampling=args.uniform_sampling)
        img_size=28
        it = iter(test_ds)
        next(it)
        next(it)
        next(it)
        next(it)
        next(it)
        #next(it)
        #next(it)
        tf.random.set_seed(10)
        (context_x, context_y, target_x), target_y = next(it)
        
        pred_y = model((context_x, context_y, target_x))

        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        # Plot context points
        blue_img = tf.tile(tf.constant([[[0.,0.,1.]]]), [28, 28, 1])
        indices = tf.cast(context_x[0] * 27., tf.int32)
        updates = tf.tile(context_y[0], [1, 3])
        context_img = tf.tensor_scatter_nd_update(blue_img, indices, updates)
        axs[0][i].imshow(context_img.numpy())
        axs[0][i].axis('off')
        axs[0][i].set_title(f'{num_context} context points')
        # Plot mean and variance
        mean = tf.tile(tf.reshape(mu[0], (28, 28, 1)), [1, 1, 3])
        var = tf.tile(tf.reshape(sigma[0], (28, 28, 1)), [1, 1, 3])
        axs[1][i].imshow(mean.numpy(), vmin=0., vmax=1.)
        axs[2][i].imshow(var.numpy(), vmin=0., vmax=1.)
        axs[1][i].axis('off')
        axs[2][i].axis('off')
        axs[1][i].set_title('Predicted mean')
        axs[2][i].set_title('Predicted variance')
#%%
train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_mnist(batch_size=BATCH_SIZE, num_context_points=num_context, uniform_sampling=args.uniform_sampling)
img_size=28
it = iter(test_ds)
next(it)
next(it)
next(it)
next(it)
#next(it)
#next(it)
#next(it)
tf.random.set_seed(10)
(context_x, context_y, target_x), target_y = next(it)
def vis(x):
    n = x.numpy().reshape(BATCH_SIZE, 28,28)[0, :, :]
    plt.imshow(np.stack((n,n,n), axis=2))
    plt.show()
vis(target_y)

#%%


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
for epoch in range(0, epochs + 1):
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

        tepoch.set_postfix({'Train loss': train_loss.result().numpy(), 'Test loss': test_loss.result().numpy()})
        
        for callback in callbacks: callback.on_epoch_end(epoch, logs={'loss': train_loss.result()})

#%%


