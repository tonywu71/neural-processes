#%%
import os
os.chdir("/Users/baker/Documents/MLMI4/conditional-neural-processes/")
import argparse
from datetime import datetime

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_probability as tfp


from dataloader.load_mnist import load_mnist, split_load_mnist
from dataloader.load_celeb import load_celeb
from model import ConditionalNeuralProcess
from utility import PlotCallback
import matplotlib.pyplot as plt
import numpy as np

tfk = tf.keras
tfd = tfp.distributions


args = argparse.Namespace(epochs=15, batch=64, task='mnist', num_context=10, uniform_sampling=True)

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs


if args.task == 'mnist':
    encoder_dims = [500, 500, 500, 500]
    decoder_dims = [500, 500, 500, 2]

    def loss(target_y, pred_y):
        # Get the distribution
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return -dist.log_prob(target_y)


model = ConditionalNeuralProcess(encoder_dims, decoder_dims)


#%%

num_context = 100
model.load_weights(f'trained_models/model_mnist_context_{num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0015.ckpt")
# Split example
it = iter(split_load_mnist(num_context))

import matplotlib.pyplot as plt
tf.random.set_seed(13)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
next(it)
(context_x, context_y, target_x), target_y, L, R = next(it)
def vis(x):
    n = x.numpy().reshape(28,28)
    plt.imshow(np.stack((n,n,n), axis=2))
    plt.show()
vis(L)
vis(R)
vis(target_y)



# print(context_x.shape)
# print(context_y.shape)
# print(target_x.shape)
# print(target_y.shape)

pred_y = model((context_x, context_y, target_x))

mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
# Plot context points
blue_img = tf.tile(tf.constant([[[0.,0.,1.]]]), [28, 28, 1])
indices = tf.cast(context_x[0] * 27., tf.int32)
updates = tf.tile(context_y[0], [1, 3])
context_img = tf.tensor_scatter_nd_update(blue_img, indices, updates)


i = 0
#fig, axs = plt.subplots(3, 1)#, figsize=(10, 5))
fig, axs = plt.subplots(3, figsize=(10, 5))

axs[0].imshow(context_img.numpy())
axs[0].axis('off')
axs[0].set_title(f'{num_context} context points')
# Plot mean and variance
mean = tf.tile(tf.reshape(mu[0], (28, 28, 1)), [1, 1, 3])
var = tf.tile(tf.reshape(sigma[0], (28, 28, 1)), [1, 1, 3])
axs[1].imshow(mean.numpy(), vmin=0., vmax=1.)
axs[2].imshow(var.numpy(), vmin=0., vmax=1.)
axs[1].axis('off')
axs[2].axis('off')
axs[1].set_title('Predicted mean')
axs[2].set_title('Predicted variance')      

# %%
