#%%
import os
os.chdir("/Users/baker/Documents/MLMI4/conditional-neural-processes/")
from utils.load_model import *
tf.config.set_visible_devices([], 'GPU') # Disable the GPU if present, we wont need it
from dataloader.load_mnist import half_load_mnist
import numpy as np

# ================================ Training parameters ===============================================

# Regression
args = argparse.Namespace(epochs=60, batch=64, task='mnist', num_context=25, uniform_sampling=True, model='HNPC')
model, train_ds, test_ds = load_model_and_dataset(args)


#%%

num_context = 100
model.load_weights(f'.data/HNPC_model_mnist_context_100_uniform_sampling_True/' + "cp-0030.ckpt")
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
# next(it)
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
