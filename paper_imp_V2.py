# %%
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial

#tf.config.set_visible_devices([], 'GPU')

# %%
from neural_process_model_v2 import NeuralProcess
from gp_curves import GPCurvesGenerator, plot_func

# %%
BATCH_SIZE = 128
TRAINING_ITERATIONS = int(100000/BATCH_SIZE) # 1e5
TEST_ITERATIONS = int(TRAINING_ITERATIONS/10)
MAX_CONTEXT_POINTS = 10
PLOT_AFTER = int(1e4)


# %%
dataset_train = GPCurvesGenerator(batch_size=BATCH_SIZE, max_size=MAX_CONTEXT_POINTS)
dataset_test = GPCurvesGenerator(batch_size=BATCH_SIZE, max_size=MAX_CONTEXT_POINTS, testing=True)
data_train = dataset_train.generate()



def gen(dataset, iterations):
    data_train = dataset.generate()
    for i in range(iterations):
        context = tf.concat((data_train[0][0], data_train[0][1]), axis=1)
        query = data_train[1]
        target = data_train[2]
        yield ((context, query), target)

train_ds = tf.data.Dataset.from_generator(
            partial(gen, dataset_train, TRAINING_ITERATIONS),
            output_types=((tf.float32, tf.float32), (tf.float32))
        )

test_ds = tf.data.Dataset.from_generator(
            partial(gen, dataset_test, TEST_ITERATIONS),
            output_types=((tf.float32, tf.float32), (tf.float32))
        )



# %%


z_output_sizes = [128, 128, 128, 128, 256]
enc_output_sizes = [128, 128, 128, 128]
dec_output_sizes = [128, 128, 2]

model = NeuralProcess(z_output_sizes, enc_output_sizes, dec_output_sizes)



#opt = tf.train.AdamOptimizer(1e-4).minimize(loss)

#%%


# def loss(target_y, pred_y):
#     mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
#     dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

#     log_prob = dist.log_prob(target_y)
#     log_prob = tf.reduce_sum(log_prob)

#     context, query = model.inputs # TODO how do we get this context & query? its from the input data 
#     prior = model.z_encoder_latent(context) # TODO can we even call methods of the model in the loss?
#     posterior = model.z_encoder_latent(tf.concat([query, target_y], axis=-1))

#     kl = tfp.distributions.kl_divergence(prior, posterior)
#     kl = tf.reduce_sum(kl)

#     # maximize variational lower bound
#     loss = -log_prob + kl
#     return loss


# def loss(target_y, pred_y):
#     mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
#     dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
#     return -dist.log_prob(target_y)


#model.compile(loss=loss, optimizer='adam')


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



time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'np', time)
writer = tf.summary.create_file_writer(log_dir)
plotter = PlotCallback(logdir=log_dir, ds=test_ds)
callbacks = [plotter]

#%%

# model.fit(train_ds, epochs=20, callbacks=callbacks)

def compute_loss(model, x):
    (context, query), target_y = x
    pred_y = model(x[0])
    mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
    dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma)

    log_prob = dist.log_prob(target_y)
    log_prob = tf.reduce_sum(log_prob)

    prior = model.z_encoder_latent(context)
    posterior = model.z_encoder_latent(tf.concat([query, target_y], axis=1))

    kl = tfp.distributions.kl_divergence(prior, posterior)
    kl = tf.reduce_sum(kl)

    # maximize variational lower bound
    loss = -log_prob + kl
    return loss


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return tf.math.reduce_mean(loss)


for callback in callbacks: callback.model = model


#%%

from tqdm import tqdm
optimizer = tf.keras.optimizers.Adam(1e-3)

epochs = 15
for epoch in range(1, epochs + 1):
    
    
    with tqdm(total=TRAINING_ITERATIONS, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        train_loss = tf.keras.metrics.Mean()
        for idx, train_x in enumerate(train_ds):
            loss = train_step(model, train_x, optimizer)
            
            train_loss(loss)
            tepoch.set_postfix({'Batch': idx, 'Train Loss': train_loss.result().numpy()})
            tepoch.update(1)
            with writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=epoch*TRAINING_ITERATIONS + idx)

            


        test_loss = tf.keras.metrics.Mean()
        for idx, test_x in enumerate(test_ds):
            loss = compute_loss(model, test_x)

            test_loss(loss)
            tepoch.set_postfix({'Batch': idx, 'Test Loss': test_loss.result().numpy()})
            with writer.as_default():
                tf.summary.scalar('test_loss', test_loss.result(), step=epoch*TEST_ITERATIONS + idx)

        tepoch.set_postfix({'Train loss': train_loss.result().numpy(), 'Test loss': test_loss.result().numpy()})
        
        for callback in callbacks: callback.on_epoch_end(epoch)

#%%
