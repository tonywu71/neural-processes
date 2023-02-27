#%%
import os
import argparse
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp

from dataloader.load_regression_data_uniform import RegressionDataGeneratorUniform
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb
from neural_process_model_v2 import NeuralProcess
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

args = argparse.Namespace(epochs=15, batch=64, task='regression', num_context=10, uniform_sampling=True)

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs


model_path = f'.data/model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-{epoch:04d}.ckpt"


data_generator = RegressionDataGeneratorUniform()
train_ds, test_ds = data_generator.load_regression_data(batch_size=BATCH_SIZE)

# Model architecture
z_output_sizes = [128, 128, 128, 128, 256]
enc_output_sizes = [128, 128, 128, 128]
dec_output_sizes = [128, 128, 2]





#%%

# Compile model
model = NeuralProcess(z_output_sizes, enc_output_sizes, dec_output_sizes)


#%%

# Callbacks
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'np', time)
writer = tf.summary.create_file_writer(log_dir)
#plotter = PlotCallback(logdir=log_dir, ds=test_ds)
plot_clbk = PlotCallback(logdir=log_dir, ds=test_ds, task=args.task)
callbacks = [plot_clbk]


#%%

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

TRAINING_ITERATIONS = int(10000) # 1e5
TEST_ITERATIONS = int(TRAINING_ITERATIONS/10)

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
