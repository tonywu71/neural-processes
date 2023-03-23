#%%
import os
os.chdir("/Users/baker/Documents/MLMI4/conditional-neural-processes/")
from utils.load_model import *
tf.config.set_visible_devices([], 'GPU') # Disable the GPU if present, we wont need it


# ================================ Training parameters ===============================================

# Regression
args = argparse.Namespace(epochs=60, batch=1024, task='regression', num_context=25, uniform_sampling=True, model='HNPC')
model, train_ds, test_ds = load_model_and_dataset(args)

if args.model == 'CNP':
    pass
elif args.model == 'HNP':
    pth = '.data/HNP_model_regression_context_25_uniform_sampling_True/'
elif args.model == 'LNP':
    pth = '.data/LNP_model_regression_context_25_uniform_sampling_True/'
elif args.model == 'HNPC':
    pth = '.data/HNPC_model_regression_context_25_uniform_sampling_True/'



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

    



import numpy as np
import plotly.graph_objects as go

def plot_distributions(dist, name=""):
    mean = np.array([np.mean(x) for x in dist])
    std = np.array([np.std(x) for x in dist])
    dmin = np.array([np.min(x) for x in dist])
    dmax = np.array([np.max(x) for x in dist])
    x = [float(list(models)[i][4:7]) for i in range(len(models))]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=dmax,
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


    fig.add_trace(go.Scatter(x=x, y=dmin,
                            mode='lines',
                            line=dict(color='#FFAAAA', width =0.1),
                            fill='tonexty',
                            name='min'))


    #fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    # fig.update_layout(title='Latent Distribution Mean', xaxis_title="Latent Distribution Mean",
    #     yaxis_title="Epoch")
    fig.update_layout(title=f'{args.model} Latent Distribution', yaxis_title=f"Latent Distribution {name}",
        xaxis_title="Epoch")
    fig.update_layout(
        #showlegend=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        # legend=dict(
        #     yanchor="top",
        #     y=0.99,
        #     xanchor="left",
        #     x=0.01
        # ),
        width=500,
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
    )

    fig.show()

plot_distributions(prior_mus, "Mean")
plot_distributions(prior_sigmas, "Standard Deviation")

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