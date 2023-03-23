
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
#tf.config.set_visible_devices([], 'GPU')
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





def load_model_and_dataset(args, model_path=None):
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

    if model_path is not None:
        if model_path.endswith(".ckpt"):
            model.load_weights(model_path)
        else:
            models = os.listdir(model_path)
            models = set([m[:12] for m in models if '.data' in m])
            models = list(models)
            models.sort()
            model.load_weights(model_path + models[-1])
    
    return model, train_ds, test_ds
