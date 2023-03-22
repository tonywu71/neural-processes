#%%
import os
os.chdir("/Users/baker/Documents/MLMI4/conditional-neural-processes/")
from utils.load_model import *


# ================================ Training parameters ===============================================

# Regression
args = argparse.Namespace(epochs=60, batch=1024, task='regression', num_context=25, uniform_sampling=True, model='HNPC')
model, train_ds, test_ds = load_model(args)

pth = f'.data/{args.model}_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0080.ckpt"

model.load_weights(pth)

#%%

BATCH_SIZE = 1


fig, axs = plt.subplots(3, 4, figsize=(10, 5))
#for i, num_context in enumerate([1,10,100,1000]):#([1,10,100,1000]):
for i, num_context in enumerate([1,10,100,1000]):#([1,10,100,1000]):

    #model.load_weights(f'trained_models/model_{args.task}_context_{num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0015.ckpt")
    #model.load_weights(f'.data/CNP2_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0010.ckpt")

    #model.load_weights(f'.data/CNP2_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0010.ckpt")
    

    if args.task == 'celeb':
        
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
        model.load_weights(f'trained_models/model_{args.task}_context_{num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-0015.ckpt")
        train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_mnist(batch_size=BATCH_SIZE, num_context_points=num_context, uniform_sampling=args.uniform_sampling)
        img_size=28
        it = iter(test_ds)
        next(it)
        next(it)
        next(it)
        next(it)
        next(it)
        next(it)
        next(it)
        tf.random.set_seed(10)
        (context_x, context_y, target_x), target_y = next(it)
        print(context_x.shape)
        print(context_y.shape)
        print(target_x.shape)
        print(target_y.shape)
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


    
if args.task == None:
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

# %%
num_context = 100



    
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3,2, figsize=(6, 5))
if args.task == 'celeb':
    for i, uniform in enumerate([True, False]):
        model.load_weights(f'trained_models/model_{args.task}_context_{num_context}_uniform_sampling_{uniform}/' + "cp-0015.ckpt")
        
        train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_celeb(batch_size=BATCH_SIZE, num_context_points=num_context, uniform_sampling=uniform)
        img_size=32

        it = iter(test_ds)
        next(it)
        # next(it)
        # next(it)
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
        axs[0][i].set_title(f'{num_context} context points ' + ("uniform" if uniform else "sorted"))
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
    it = iter(test_ds)
    next(it)
    next(it)
    next(it)
    next(it)
    next(it)
    next(it)
    next(it)
    tf.random.set_seed(10)
    
    (context_x, context_y, target_x), target_y = next(it)
    for i, uniform in enumerate([True, False]):
        model.load_weights(f'trained_models/model_{args.task}_context_{num_context}_uniform_sampling_{uniform}/' + "cp-0015.ckpt")
        train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_mnist(batch_size=BATCH_SIZE, num_context_points=num_context, uniform_sampling=uniform)
        img_size=28
        
        pred_y = model((context_x, context_y, target_x))

        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        # Plot context points
        blue_img = tf.tile(tf.constant([[[0.,0.,1.]]]), [28, 28, 1])
        indices = tf.cast(context_x[0] * 27., tf.int32)
        updates = tf.tile(context_y[0], [1, 3])
        context_img = tf.tensor_scatter_nd_update(blue_img, indices, updates)
        axs[0][i].imshow(context_img.numpy())
        axs[0][i].axis('off')
        axs[0][i].set_title(f'{num_context} context points ' + ("uniform" if uniform else "sorted"))
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