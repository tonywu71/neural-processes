#%%
import os
os.chdir("c:Users/baker/Documents/MLMI4/conditional-neural-processes/")
import argparse
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('example-cnp/')

from dataloader.load_regression_data_from_arbitrary_gp import RegressionDataGeneratorArbitraryGP
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb

from neural_process_model_hybrid import NeuralProcessHybrid
from neural_process_model_latent import NeuralProcessLatent
#from nueral_process_model_conditional import ConditionalNeuralProcess as NeuralProcess
from utils.utility import PlotCallback

tfk = tf.keras
tfd = tfp.distributions

# # Parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--epochs', type=int, default=15, help='Number of training epochs')
# parser.add_argument('-b', '--batch', type=int, default=64, help='Batch size for training')
# parser.add_argument('-t', '--task', type=str, default='mnist', help='Task to perform : (mnist|regression)')

# args = parser.parse_args()

tf.config.set_visible_devices([], 'GPU') # DONT use the GPU, not needed
args = argparse.Namespace(epochs=15, batch=2, task='regression', num_context=10, uniform_sampling=True, model='HNP')


# Training parameters
BATCH_SIZE = args.batch
EPOCHS = args.epochs



TRAINING_ITERATIONS = int(100) # 1e5
TEST_ITERATIONS = int(TRAINING_ITERATIONS/5)
if args.task == 'mnist':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_mnist(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)
    
    # Model architecture
    z_output_sizes = [500, 500, 500, 500]
    enc_output_sizes = [500, 500, 500, 1000]
    dec_output_sizes = [500, 500, 500, 2]


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
        kernel_length_scale=0.4,
        
    )
    train_ds, test_ds = data_generator.load_regression_data()

    # Model architecture
    z_output_sizes = [500, 500, 500, 500]
    enc_output_sizes = [500, 500, 500, 1000]
    dec_output_sizes = [500, 500, 500, 2]

    

elif args.task == 'celeb':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_celeb(batch_size=BATCH_SIZE, num_context_points=args.num_context,
                                   uniform_sampling=args.uniform_sampling)

    # Model architecture
    z_output_sizes = [500, 500, 500, 500]
    enc_output_sizes = [500, 500, 500, 1000]
    dec_output_sizes = [500, 500, 500, 6]









#%%

# Define NP Model
if args.model == 'LNP':
    model = NeuralProcessLatent(z_output_sizes, enc_output_sizes, dec_output_sizes)
elif args.model == 'HNP':
    model = NeuralProcessHybrid(z_output_sizes, enc_output_sizes, dec_output_sizes)




#%%
def plot_mean_with_std(x: np.ndarray,
                       mean: np.ndarray,
                       std: np.ndarray,
                       y_true=None,
                       ax=None,
                       alpha: float=0.3) -> plt.Axes:
    """Plot mean and standard deviation.
    IMPORTANT: x should be sorted and mean and std should be sorted in the same order as x."""
    
    if ax is None:
        ax = plt.gca()
    
    # Plot mean:
    ax.plot(x, mean, label="Mean prediction")  # type: ignore
    
    # Plot target if given:
    if y_true is not None:
        ax.plot(x, y_true, label="True function")
    
    # Plot standard deviation:
    ax.fill_between(x,  # type: ignore
                    mean - std,  # type: ignore
                    mean + std,  # type: ignore
                    alpha=alpha,
                    label="Standard deviation")
    
    ax.legend()
    
    return ax  # type: ignore
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


#%%


import os
pth = '.data/HNP_model_regression_context_10_uniform_sampling_True/'
# pth = '.data/LNP_model_regression_context_10_uniform_sampling_True/'
models = os.listdir(pth)
models = set([m[:12] for m in models if '.data' in m])
models = list(models)
models.sort()
#models.remove('cp-0061.ckpt')

tf.random.set_seed(3)
test_iter = iter(test_ds)
x = next(test_iter)

prior_mus = []
prior_sigmas = []

posterior_mus = []
posterior_sigmas = []
for name in models:
    model_path = pth
    model.load_weights(model_path + name)
    (context_x, context_y, target_x), target_y = x
    context = tf.concat((context_x, context_y), axis=-1)
    hidden = model.z_encoder_latent.model(context)

    mu, log_sigma = tf.split(hidden, num_or_size_splits=2, axis=-1) # split the output in half
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
    prior_mus.append(mu.numpy().reshape(-1))
    prior_sigmas.append(sigma.numpy().reshape(-1))

    target_context = tf.concat((target_x, target_y), axis=2)
    hidden = model.z_encoder_latent.model(target_context)
    mu, log_sigma = tf.split(hidden, num_or_size_splits=2, axis=-1) # split the output in half
    sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
    posterior_mus.append(mu.numpy().reshape(-1))
    posterior_sigmas.append(sigma.numpy().reshape(-1))

    


#%%

import plotly.graph_objects as go
from plotly.colors import n_colors
import numpy as np



data = []
# for i in range(12):
#     r = np.random.random((1024, 128))
#     #dist = np.histogram(r, bins=np.linspace(np.min(r), np.max(r), 100))
#     #dist = (dist[0] / np.sum(dist[0]), dist[1])
#     data.append(r.reshape(-1))

#data = zip(prior_mus, posterior_mus)
data = zip(prior_sigmas, posterior_sigmas)

colors = zip(n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb'), n_colors('rgb(200, 200, 5)', 'rgb(10, 200, 10)', 12, colortype='rgb'))

fig = go.Figure()
i = 0
for data_line, color in zip(data, colors):
    prior_mu, posterior_mu = data_line
    col_a, col_b = color
    y = float(list(models)[i][5:7])
    fig.add_trace(go.Violin(x=prior_mu, line_color=col_a, name=y))#list(models)[i][5:7]))
    fig.add_trace(go.Violin(x=posterior_mu, line_color=col_b, name=y+0.1))#list(models)[i][5:7]))
    i+=1

fig.update_traces(orientation='h', side='positive', width=3, points=False)
fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
# fig.update_layout(title='Latent Distribution Mean', xaxis_title="Latent Distribution Mean",
#     yaxis_title="Epoch")
fig.update_layout(title='HNP Prior vs Posterior Latent Distribution', xaxis_title="Latent Distribution Standard Deviation",
    yaxis_title="Epoch")
fig.update_layout(
    showlegend=False,
    width=400,
    height=300,
    margin=dict(l=20, r=20, t=30, b=20),
)
fig.show()

#%%