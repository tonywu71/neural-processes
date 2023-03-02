#%%
import os
import argparse
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp

import sys
sys.path.append('example-cnp/')

from dataloader.load_regression_data_from_arbitrary_gp import RegressionDataGeneratorArbitraryGP
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb

from neural_process_model import NeuralProcess
#from model import ConditionalNeuralProcess as NeuralProcess
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

args = argparse.Namespace(epochs=15, batch=1024, task='regression', num_context=10, uniform_sampling=True)

# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs


model_path = f'.data/model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-{epoch:04d}.ckpt"

TRAINING_ITERATIONS = int(100) # 1e5
TEST_ITERATIONS = TRAINING_ITERATIONS
if args.task == 'mnist':
    train_ds, test_ds = load_mnist(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)
    
    # Model architecture
    z_output_sizes = [500, 500, 500, 500]
    enc_output_sizes = [500, 500, 500, 1000]
    dec_output_sizes = [500, 500, 500, 2]


elif args.task == 'regression':
    data_generator = RegressionDataGeneratorArbitraryGP(
        iterations=TRAINING_ITERATIONS,
        batch_size=BATCH_SIZE,
        min_num_context=3,
        max_num_context=40,
        min_num_target=2,
        max_num_target=40,
        min_x_val_uniform=-2,
        max_x_val_uniform=2,
        kernel_length_scale=0.4
    )
    train_ds, test_ds = data_generator.load_regression_data()
    # train_ds = train_ds.prefetch(5)
    # test_ds = test_ds.prefetch(5)

    # Model architecture
    z_output_sizes = [500, 500, 500, 500]
    enc_output_sizes = [500, 500, 500, 1000]
    dec_output_sizes = [500, 500, 500, 2]

    # z_output_sizes = [128, 128, 128, 128, 256]
    # enc_output_sizes = [128, 128, 128, 128]
    # dec_output_sizes = [128, 128, 2]
    

    

elif args.task == 'celeb':
    train_ds, test_ds = load_celeb(batch_size=BATCH_SIZE, num_context_points=args.num_context,
                                   uniform_sampling=args.uniform_sampling)

    # Model architecture
    z_output_sizes = [500, 500, 500, 500]
    enc_output_sizes = [500, 500, 500, 1000]
    dec_output_sizes = [500, 500, 500, 2]









#%%

# Compile model
model = NeuralProcess(z_output_sizes, enc_output_sizes, dec_output_sizes)


#%%

# Callbacks
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', 'np', args.task, time)
writer = tf.summary.create_file_writer(log_dir)
# tensorboard_clbk = tfk.callbacks.TensorBoard(
#     log_dir=log_dir, update_freq='batch')
plot_clbk = PlotCallback(logdir=log_dir, ds=test_ds, task=args.task)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True)
callbacks = [plot_clbk, cp_callback]


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


for callback in callbacks: callback.model = model


#%%

from tqdm import tqdm
optimizer = tf.keras.optimizers.Adam(1e-3)



#%%

epochs = 60
for epoch in range(1, epochs + 1):
    with tqdm(total=TRAINING_ITERATIONS, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        train_loss = tf.keras.metrics.Mean()
        for idx, train_x in enumerate(train_ds):

            # if idx == 5:
            #     tf.profiler.experimental.start(log_dir)
            #     print("profiling!")
            #tf.summary.trace_on(graph=True) # Uncomment to trace the computational graph

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


def loss(target_y, pred_y):
    # Get the distribution
    mu, sigma = tf.split(pred_y, num_or_size_splits=2, axis=-1)
    dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    return -dist.log_prob(target_y)

tensorboard_clbk = tfk.callbacks.TensorBoard(
    log_dir=log_dir, update_freq='batch')#, profile_batch = '500,520')
callbacks.append(tensorboard_clbk)

opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=loss, optimizer=opt)

# %%

model.fit(train_ds, epochs=EPOCHS, callbacks=callbacks)

#%%

i = 0
it = iter(test_ds)
while True:
    i += 1
    # for x in test_ds:
    
    x = next(it, None)
    if x == None:
        it = iter(test_ds)
        x = next(it)
    (context_x, context_y, query), target_y = x
    x_shape = tf.shape(context_x)
    y_shape = tf.shape(context_y)
    q_shape = tf.shape(query)
    t_shape = tf.shape(target_y)

    target_context = tf.concat([context_x, context_y], axis=-1)
    ts = tf.shape(target_context)

    if (x_shape[1] != y_shape[1]) or (ts[-1] != 2) or (q_shape[1] != t_shape[1]):
        print(f"x:{x_shape}     y:{y_shape}      i={i}      t: {ts}     {q_shape}       {t_shape}")
#%%