#%%
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
#tf.config.set_visible_devices([], 'GPU')
import sys
sys.path.append('example-cnp/')
from dataloader.load_regression_data_from_arbitrary_gp import RegressionDataGeneratorArbitraryGP
from dataloader.load_regression_data_from_arbitrary_gp_varying_kernel import RegressionDataGeneratorArbitraryGPWithVaryingKernel
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb
from neural_process_model_conditional import NeuralProcessConditional
from neural_process_model_hybrid import NeuralProcessHybrid
from neural_process_model_latent import NeuralProcessLatent
from neural_process_model_hybrid_constrained import NeuralProcessHybridConstrained
from utils.utility import PlotCallback

tfk = tf.keras
tfd = tfp.distributions

# Parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--epochs', type=int, default=120, help='Number of training epochs')
# parser.add_argument('-b', '--batch', type=int, default=1024, help='Batch size for training')
# parser.add_argument('-t', '--task', type=str, default='regression', help='Task to perform : (mnist|regression)')
# parser.add_argument('-c', '--num_context', type=int, default=100)
# parser.add_argument('-u', '--uniform_sampling', type=bool, default=True)
# args = parser.parse_args()







# ================================ Training parameters ===============================================

# Regression
args = argparse.Namespace(epochs=80, batch=1024, task='regression_varying', num_context=25, uniform_sampling=True, model='LNP')

# MNIST / Celeb
#args = argparse.Namespace(epochs=30, batch=256, task='mnist', num_context=100, uniform_sampling=True, model='HNPC')

LOG_PRIORS = True

# -------------------------------------------------------------------------------------------------------------------------






# =========================== Data Loaders ===========================================================================================
BATCH_SIZE = args.batch
EPOCHS = args.epochs
TRAINING_ITERATIONS = int(100)
TEST_ITERATIONS = int(TRAINING_ITERATIONS/5)
if args.task == 'mnist':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_mnist(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)

    # Model architecture
    z_output_sizes = [500, 500, 500, 1000]
    enc_output_sizes = [500, 500, 500, 500]
    dec_output_sizes = [500, 500, 500, 2]

elif args.task == 'celeb':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_celeb(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)

    # Model architecture
    z_output_sizes = [500, 500, 500, 1000]
    enc_output_sizes = [500, 500, 500, 500]
    dec_output_sizes = [500, 500, 500, 6]

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
        kernel_length_scale=0.4
    )
    train_ds, test_ds = data_generator.load_regression_data()

    # Model architecture
    z_output_sizes = [500, 500, 500, 1000]
    enc_output_sizes = [500, 500, 500, 500]
    dec_output_sizes = [500, 500, 500, 2]    

elif args.task == 'regression_varying':
    data_generator = RegressionDataGeneratorArbitraryGPWithVaryingKernel(
        iterations=TRAINING_ITERATIONS,
        n_iterations_test=TEST_ITERATIONS,
        batch_size=BATCH_SIZE,
        min_num_context=3,
        max_num_context=40,
        min_num_target=2,
        max_num_target=40,
        min_x_val_uniform=-2,
        max_x_val_uniform=2,
        min_kernel_length_scale=0.1,
        max_kernel_length_scale=1.
    )
    # Model architecture
    z_output_sizes = [500, 500, 500, 1000]
    enc_output_sizes = [500, 500, 500, 500]
    dec_output_sizes = [500, 500, 500, 2]

    train_ds, test_ds = data_generator.train_ds, data_generator.test_ds
    



# --------------------------------------------------------------------------------------------------------------------------------------------





# ========================================== Define NP Model ===================================================
if args.model == 'CNP':
    model = NeuralProcessConditional(enc_output_sizes, dec_output_sizes)
elif args.model == 'HNP':
    model = NeuralProcessHybrid(z_output_sizes, enc_output_sizes, dec_output_sizes)
elif args.model == 'LNP':
    model = NeuralProcessLatent(z_output_sizes, enc_output_sizes, dec_output_sizes)
elif args.model == 'HNPC':
    model = NeuralProcessHybridConstrained(z_output_sizes, enc_output_sizes, dec_output_sizes)
    

optimizer = tf.keras.optimizers.Adam(1e-3)

# Callbacks
model_path = f'.data/{args.model}_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/' + "cp-{epoch:04d}.ckpt"
time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('.', 'logs', args.model, args.task, time)
writer = tf.summary.create_file_writer(log_dir)
plot_clbk = PlotCallback(logdir=log_dir, ds=test_ds, task=args.task)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True)
callbacks = [plot_clbk, cp_callback]
for callback in callbacks: callback.model = model

if args.model == 'HNPC':
    model.writer = writer

# -----------------------------------------------------------------------------------------------------------



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

# ============================ Training Loop ===========================================================
epochs = args.epochs
epochs = 160
for epoch in range(1, epochs + 1):
    with tqdm(total=TRAINING_ITERATIONS, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        # ------------------------------- Training --------------------------------------------
        train_loss = tf.keras.metrics.Mean()
        for idx, train_x in enumerate(train_ds):

            loss = train_step(model, train_x, optimizer)



            # VVVVVVVVVVVVVVVVVVV Logging VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
            train_loss(loss)
            tepoch.set_postfix({'Batch': idx, 'Train Loss': train_loss.result().numpy()})
            tepoch.update(1)
            with writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=epoch*TRAINING_ITERATIONS + idx)


            


        # ------------------------------ Testing -----------------------------------------------
        test_loss = tf.keras.metrics.Mean()
        for idx, test_x in enumerate(test_ds):

            loss = model.compute_loss(test_x)
            

            # VVVVVVVVVVVVVVVVVVV Logging VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
            test_loss(loss)
            tepoch.set_postfix({'Batch': idx, 'Test Loss': test_loss.result().numpy()})
            with writer.as_default():
                tf.summary.scalar('test_loss', test_loss.result(), step=epoch*TEST_ITERATIONS + idx)





        # ---------------------- Logging Prior & Posterior Distribution -----------------------------------
        if LOG_PRIORS and args.model != 'CNP':
            (context_x, context_y, target_x), target_y = next(iter(test_ds))
            
            prior_context = tf.concat((context_x, context_y), axis=2)
            prior = model.z_encoder_latent(prior_context)

            target_context = tf.concat((target_x, target_y), axis=2)
            posterior = model.z_encoder_latent(target_context)

            with writer.as_default():
                tf.summary.histogram('prior/mu', prior.mean(), epoch)
                tf.summary.histogram('prior/sigma', prior.stddev(), epoch)
                tf.summary.histogram('posterior/mu', posterior.mean(), epoch)
                tf.summary.histogram('posterior/sigma', posterior.stddev(), epoch)



        # ------------- Some callbacks  -----------------------------------------------------------------------
        tepoch.set_postfix({'Train loss': train_loss.result().numpy(), 'Test loss': test_loss.result().numpy()})
        for callback in callbacks: callback.on_epoch_end(epoch, logs={'loss': train_loss.result()})
        writer.flush()

#%%


