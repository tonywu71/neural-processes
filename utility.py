import io

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras

import matplotlib.pyplot as plt

from utils.gaussian_processes.plot_gp_utils import plot_mean_with_std



def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_regression(target_x, target_y, context_x, context_y, pred_y):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    
    x_val = tf.squeeze(target_x[0, :, :])
    pred_y = tf.squeeze(pred_y[0, :, :])
    y_true = tf.squeeze(target_y[0, :, :])
    
    idx_x_sorted = tf.argsort(x_val)
    
    x_val = tf.gather(x_val, idx_x_sorted)
    mean = tf.gather(pred_y[:, 0], idx_x_sorted)
    std = tf.gather(pred_y[:, 1], idx_x_sorted)
    y_true = tf.gather(y_true, idx_x_sorted)
    
    plot_mean_with_std(x=x_val, mean=mean, std=std, y_true=y_true, ax=ax)
    ax.scatter(context_x[0, :, :], context_y[0, :, :])  # type: ignore
    
    return fig


def plot_image(target_x, target_y, context_x, context_y, pred_y):
    mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    # Plot context points
    blue_img = tf.tile(tf.constant([[[0.,0.,1.]]]), [28, 28, 1])
    indices = tf.cast(context_x[0] * 27., tf.int32)
    updates = tf.tile(context_y[0], [1, 3])
    context_img = tf.tensor_scatter_nd_update(blue_img, indices, updates)
    axes[0].imshow(context_img.numpy())
    axes[0].axis('off')
    axes[0].set_title('Given context')
    # Plot mean and variance
    mean = tf.tile(tf.reshape(mu[0], (28, 28, 1)), [1, 1, 3])
    var = tf.tile(tf.reshape(sigma[0], (28, 28, 1)), [1, 1, 3])
    axes[1].imshow(mean.numpy(), vmin=0., vmax=1.)
    axes[2].imshow(var.numpy(), vmin=0., vmax=1.)
    axes[1].axis('off')
    axes[2].axis('off')
    axes[1].set_title('Predicted mean')
    axes[2].set_title('Predicted variance')
    return fig


def plot_image_celeb(target_x, target_y, context_x, context_y, pred_y, img_size=32):
	mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
	# Plot context points
	white_img = tf.tile(tf.constant([[[0.,0.,0.]]]), [img_size, img_size, 1])
	indices = tf.cast(context_x[0] * float(img_size - 1.0), tf.int32)

	updates = context_y[0]

	context_img = tf.tensor_scatter_nd_update(white_img, indices, updates)
	axes[0].imshow(context_img.numpy())
	axes[0].axis('off')
	axes[0].set_title('Given context')
	# Plot mean and variance
	mean = tf.reshape(mu[0], (img_size, img_size, 3))
	var = tf.reshape(sigma[0], (img_size, img_size, 3))
	axes[1].imshow(mean.numpy(), vmin=0., vmax=1.)
	axes[2].imshow(var.numpy(), vmin=0., vmax=1.)
	axes[1].axis('off')
	axes[2].axis('off')
	axes[1].set_title('Predicted mean')
	axes[2].set_title('Predicted variance')
	return fig


class PlotCallback(tfk.callbacks.Callback):
    def __init__(self, logdir, ds, task):
        super(PlotCallback, self).__init__()
        self.ds = iter(ds)
        logdir += '/plots'
        self.file_writer = tf.summary.create_file_writer(logdir=logdir)
        if task == 'mnist':
            self.plot_fn = plot_image
        elif task == 'celeb':
            self.plot_fn = plot_image_celeb
        else:
            self.plot_fn = plot_regression
        self.test_ds = ds
        self.test_it = iter(self.test_ds)


    def get_next_data(self):
        try:
            next_data = next(self.test_it)
        except StopIteration:
            self.test_it = iter(self.test_ds)
            next_data = next(self.test_it)
        return next_data


    def on_epoch_end(self, epoch, logs=None):
        (context_x, context_y, target_x), target_y = self.get_next_data()
        pred_y = self.model((context_x, context_y, target_x))
        fig = self.plot_fn(target_x, target_y, context_x, context_y, pred_y)
        fig.suptitle(f'loss {logs["loss"]:.5f}')
        img = plot_to_image(fig)
        with self.file_writer.as_default():
            tf.summary.image(name="CNP image completion", data=img, step=epoch)
