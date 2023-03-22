#%%
import os
os.chdir("c:Users/baker/Documents/MLMI4/conditional-neural-processes/")
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
from dataloader.load_mnist import load_mnist
from dataloader.load_celeb import load_celeb
from nueral_process_model_conditional import NeuralProcessConditional
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
args = argparse.Namespace(epochs=60, batch=1024, task='regression', num_context=1000, uniform_sampling=True, model='LNP')

# MNIST / Celeb
#args = argparse.Namespace(epochs=30, batch=256, task='celeb', num_context=100, uniform_sampling=True, model='CNP')

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


elif args.task == 'celeb':
    train_ds, test_ds, TRAINING_ITERATIONS, TEST_ITERATIONS = load_celeb(batch_size=BATCH_SIZE, num_context_points=args.num_context, uniform_sampling=args.uniform_sampling)

    # Model architecture
    z_output_sizes = [500, 500, 500, 1000]
    enc_output_sizes = [500, 500, 500, 500]
    dec_output_sizes = [500, 500, 500, 6]

# --------------------------------------------------------------------------------------------------------------------------------------------





# ========================================== Define NP Model ===================================================
if args.model == 'CNP':
    model = NeuralProcessConditional(enc_output_sizes, dec_output_sizes)
elif args.model == 'HNP':
    model = NeuralProcessHybrid(z_output_sizes, enc_output_sizes, dec_output_sizes)
    pth = '.data/HNP_model_regression_context_25_uniform_sampling_True/'
elif args.model == 'LNP':
    model = NeuralProcessLatent(z_output_sizes, enc_output_sizes, dec_output_sizes)
    pth = '.data/LNP_model_regression_context_25_uniform_sampling_True/'
elif args.model == 'HNPC':
    model = NeuralProcessHybridConstrained(z_output_sizes, enc_output_sizes, dec_output_sizes)

#%%




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
            
    prior_context = tf.concat((context_x, context_y), axis=2)
    prior = model.z_encoder_latent(prior_context)

    target_context = tf.concat((target_x, target_y), axis=2)
    posterior = model.z_encoder_latent(target_context)

    prior_mus.append(prior.mean().numpy().reshape(-1))
    prior_sigmas.append(prior.stddev().numpy().reshape(-1))
    posterior_mus.append(posterior.mean().numpy().reshape(-1))
    posterior_sigmas.append(posterior.stddev().numpy().reshape(-1))

    


#%%
import numpy as np
import plotly.graph_objects as go

def plot_distributions(dist):
    mean = np.array([np.mean(x) for x in dist])
    std = np.array([np.std(x) for x in dist])
    min = np.array([np.min(x) for x in dist])
    max = np.array([np.max(x) for x in dist])
    x = [float(list(models)[i][5:7]) for i in range(len(models))]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=max,
                                        mode='lines',
                                        line=dict(color='#FFAAAA',width =0.1),
                                        name='max'))

    fig.add_trace(go.Scatter(x=x, y=mean+std,
                                        mode='lines',
                                        line=dict(color='#FFAAAA',width =0.1),
                                        fill='tonexty',
                                        name='upper bound'))

    fig.add_trace(go.Scatter(x=x, y=mean,
                            mode='lines',
                            line=dict(color='#FF0000'),
                            fill='tonexty',
                            name='mean'))

    fig.add_trace(go.Scatter(x=x, y=mean-std,
                            mode='lines',
                            line=dict(color='#FF0000', width =0.1),
                            fill='tonexty',
                            name='lower bound'))


    fig.add_trace(go.Scatter(x=x, y=min,
                            mode='lines',
                            line=dict(color='#FFAAAA', width =0.1),
                            fill='tonexty',
                            name='min'))


    #fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    # fig.update_layout(title='Latent Distribution Mean', xaxis_title="Latent Distribution Mean",
    #     yaxis_title="Epoch")
    fig.update_layout(title='HNP Latent Distribution', yaxis_title="Latent Distribution Standard Deviation",
        xaxis_title="Epoch")
    fig.update_layout(
        #showlegend=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        width=400,
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
    )

    fig.show()

plot_distributions(prior_mus)
#plot_distributions(prior_sigmas)

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