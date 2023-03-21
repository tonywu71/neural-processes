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
from dataloader.load_mnist import half_load_mnist
from dataloader.load_celeb import load_celeb

from neural_process_model_hybrid import NeuralProcessHybrid
from neural_process_model_latent import NeuralProcessLatent
from utility import PlotCallback

import matplotlib.pyplot as plt

tfk = tf.keras
tfd = tfp.distributions





args = argparse.Namespace(epochs=60, batch=64, task='mnist', num_context=10, uniform_sampling=True, model='LNP')


BATCH_SIZE = args.batch
EPOCHS = args.epochs


model_path = f'.data/{args.model}_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-{epoch:04d}.ckpt"

TRAINING_ITERATIONS = int(100) # 1e5
TEST_ITERATIONS = int(TRAINING_ITERATIONS/5)

z_output_sizes = [500, 500, 500, 1000]
enc_output_sizes = [500, 500, 500, 500]
dec_output_sizes = [500, 500, 500, 2]








# Define NP Model
if args.model == 'LNP':
    model = NeuralProcessLatent(z_output_sizes, enc_output_sizes, dec_output_sizes)
elif args.model == 'HNP':
    model = NeuralProcessHybrid(z_output_sizes, enc_output_sizes, dec_output_sizes)


#%%
import numpy as np

num_context = 100
#model.load_weights(f'trained_models/model_mnist_context_{num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0015.ckpt")
model.load_weights('.data/LNP_model_mnist_context_100_uniform_sampling_True/cp-0092.ckpt')
# Split example
num_context = 50
it = iter(half_load_mnist(num_context))

import matplotlib.pyplot as plt
tf.random.set_seed(13)
next(it)
next(it)
next(it)
next(it)
# next(it)
# next(it)
# next(it)
# next(it)
# next(it)
# next(it)
# next(it) #
# next(it)
# next(it)
# next(it)
(context_x, context_y, target_x), target_y= next(it)
def vis(x):
    n = x.numpy().reshape(28,28)
    plt.imshow(np.stack((n,n,n), axis=2))
    plt.show()
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
