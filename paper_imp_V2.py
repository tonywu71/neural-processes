# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial

tf.config.set_visible_devices([], 'GPU')

# %%
from neural_process_model_v2 import NeuralProcess
from gp_curves import GPCurvesGenerator, plot_func

# %%
TRAINING_ITERATIONS = int(500) # 1e5
MAX_CONTEXT_POINTS = 10
PLOT_AFTER = int(1e4)

# %%
dataset_train = GPCurvesGenerator(batch_size=64, max_size=MAX_CONTEXT_POINTS)
dataset_test = GPCurvesGenerator(batch_size=1, max_size=MAX_CONTEXT_POINTS, testing=True)
data_train = dataset_train.generate()



def gen(dataset):
    data_train = dataset.generate()
    for i in range(TRAINING_ITERATIONS):
        context = tf.concat((data_train[0][0], data_train[0][1]), axis=1)
        query = data_train[1]
        target = data_train[2]
        yield ((context, query), target)

train_ds = tf.data.Dataset.from_generator(
            partial(gen, dataset_train),
            output_types=((tf.float32, tf.float32), tf.float32)
        )

test_ds = tf.data.Dataset.from_generator(
            partial(gen, dataset_test),
            output_types=((tf.float32, tf.float32), tf.float32)
        )



# %%


z_output_sizes = [128, 128, 128, 128, 256]
enc_output_sizes = [128, 128, 128, 128]
dec_output_sizes = [128, 128, 2]

model = NeuralProcess(z_output_sizes, enc_output_sizes, dec_output_sizes)



#opt = tf.train.AdamOptimizer(1e-4).minimize(loss)

#%%
# def loss(target_y, pred_y):
#     dist, _, _ = pred_y#self(context, query)
#     log_prob = dist.log_prob(target)
#     log_prob = tf.reduce_sum(log_prob)

#     prior, _, _ = self.z_prob(self.z_encoder(context))
#     posterior, _, _ = self.z_prob(self.z_encoder([query, target]))

#     kl = tfp.distributions.kl_divergence(prior, posterior)
#     kl = tf.reduce_sum(kl)

#     # maximize variational lower bound
#     loss = -log_prob + kl
#     return loss


def loss(target_y, pred_y):
    # Get the distribution
    mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
    #dist,mu, sigma = pred_y
    # mu = pred_y[1]
    # sigma = pred_y[2]
    
    dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    #dist = pred_y
    return -dist.log_prob(target_y)

model.compile(loss=loss, optimizer='adam')


# %%



import io
class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir, ds):
        super(PlotCallback, self).__init__()
        self.ds = iter(ds)
        logdir += '/plots'
        self.file_writer = tf.summary.create_file_writer(logdir=logdir)

        self.test_ds = ds
        self.test_it = iter(self.test_ds)

    def plot(self):
        test_sample = next(self.test_it)
        pred_y = self.model(test_sample[0])
        mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
        
        (context, query), target = test_sample
        cx, cy = tf.split(context, num_or_size_splits=2, axis=1)
        return plot_func(query, target, cx, cy, mu, sigma)

    def get_next_data(self):
        try:
            next_data = next(self.test_it)
        except StopIteration:
            self.test_it = iter(self.test_ds)
            next_data = next(self.test_it)
        return next_data
    
    def plot_to_image(figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        #plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


    def on_epoch_end(self, epoch, logs=None):
        fig = self.plot()
        img = PlotCallback.plot_to_image(fig)
        with self.file_writer.as_default():
            tf.summary.image(name="NP image completion", data=img, step=epoch)


plotter = PlotCallback('np_logs/', test_ds)

callbacks = [plotter]

#%%

model.fit(train_ds, epochs=20, callbacks=callbacks)





#%%




