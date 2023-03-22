#%%
import os
os.chdir("/Users/baker/Documents/MLMI4/conditional-neural-processes/")
from utils.load_model import *
tf.config.set_visible_devices([], 'GPU') # Disable the GPU if present, we wont need it
from dataloader.load_regression_data_from_arbitrary_gp_varying_kernel import RegressionDataGeneratorArbitraryGPWithVaryingKernel
from utils.gaussian_processes.gp_model import plot_mean_with_std
from utils.gaussian_processes.plot_gp_utils import plot_preds_from_ds_test
import numpy as np
import matplotlib.pyplot as plt
from dataloader.load_regression_data_from_arbitrary_gp_varying_kernel import draw_single_example_from_arbitrary_gp
from utils.gaussian_processes.plot_gp_utils import plot_preds_from_single_example



args = argparse.Namespace(epochs=60, batch=1024, task='regression_varying', num_context=25, uniform_sampling=True, model='CNP')
model, train_ds, test_ds = load_model_and_dataset(args, ".data/CNP_model_regression_varying_context_25_uniform_sampling_True/cp-0080.ckpt")
args = argparse.Namespace(epochs=60, batch=1024, task='regression_varying', num_context=25, uniform_sampling=True, model='LNP')
lnp_model, _, _ = load_model_and_dataset(args, ".data/LNP_model_regression_varying_context_25_uniform_sampling_True/cp-0080.ckpt")
args = argparse.Namespace(epochs=60, batch=1024, task='regression_varying', num_context=25, uniform_sampling=True, model='HNPC')
hnpc_model, _, _ = load_model_and_dataset(args, ".data/HNPC_model_regression_varying_context_25_uniform_sampling_True/cp-0080.ckpt")


# %%
(context_x, context_y, target_x), target_y = next(iter(train_ds))


# %%
for (context_x, context_y, target_x), target_y in train_ds.take(5):
    RegressionDataGeneratorArbitraryGPWithVaryingKernel.plot_first_elt_of_batch(context_x, context_y, target_x, target_y)


# %%
fig = plot_preds_from_ds_test(model, ds_test=test_ds, num_samples=1)

# %%
fig = plot_preds_from_ds_test(model, ds_test=test_ds, num_samples=5)

# %%



list_num_context = [10, 20, 50]
list_kernel_length_scale = [0.3, 0.6, 0.9]

fig, axis = plt.subplots(len(list_num_context), len(list_kernel_length_scale),
                         figsize=(4*len(list_kernel_length_scale), 3*len(list_num_context)),
                         sharex=True,
                         sharey=True)

for idx_row, num_context in enumerate(list_num_context):
    for idx_col, kernel_length_scale in enumerate(list_kernel_length_scale):
        (context_x, context_y, target_x), target_y = draw_single_example_from_arbitrary_gp(
            kernel_length_scale=kernel_length_scale,
            num_context=num_context,
            num_target=100
        )
    
        plot_preds_from_single_example(model, context_x, context_y, target_x, target_y,
                                       show_title=False, ax=axis[idx_row, idx_col])

        
for ax, label in zip(axis[:,0], [f"n_context = {n}" for n in list_num_context]):
    ax.set_ylabel(label, rotation=90, size='large')
        
for ax, label in zip(axis[-1,:], [f"l = {l}" for l in list_kernel_length_scale]):
    ax.set_xlabel(label, rotation=0, size='large')

fig.tight_layout()

#%%

list_num_context = [10, 20, 50]
list_kernel_length_scale = [0.3, 0.6, 0.9]

fig, axis = plt.subplots(len(list_num_context), len(list_kernel_length_scale),
                         figsize=(4*len(list_kernel_length_scale), 3*len(list_num_context)),
                         sharex=True,
                         sharey=True)

for idx_row, num_context in enumerate(list_num_context):
    for idx_col, kernel_length_scale in enumerate(list_kernel_length_scale):
        (context_x, context_y, target_x), target_y = draw_single_example_from_arbitrary_gp(
            kernel_length_scale=kernel_length_scale,
            num_context=num_context,
            num_target=100
        )
    
        plot_preds_from_single_example(lnp_model, context_x, context_y, target_x, target_y,
                                       show_title=False, ax=axis[idx_row, idx_col])

        
for ax, label in zip(axis[:,0], [f"n_context = {n}" for n in list_num_context]):
    ax.set_ylabel(label, rotation=90, size='large')
        
for ax, label in zip(axis[-1,:], [f"l = {l}" for l in list_kernel_length_scale]):
    ax.set_xlabel(label, rotation=0, size='large')

fig.tight_layout()

# %%
num_context = 15
list_kernel_length_scale = [0.3, 0.6, 0.9]

fig, axis = plt.subplots(2, len(list_kernel_length_scale),
                         figsize=(4*len(list_kernel_length_scale), 5),
                         sharex=True,
                         sharey=True)

for idx_col, kernel_length_scale in enumerate(list_kernel_length_scale):
    (context_x, context_y, target_x), target_y = draw_single_example_from_arbitrary_gp(
        kernel_length_scale=kernel_length_scale,
        num_context=num_context,
        num_target=100
    )
    
    plot_preds_from_single_example(model, context_x, context_y, target_x, target_y,
                                   show_title=False, ax=axis[0, idx_col])

    plot_preds_from_single_example(lnp_model, context_x, context_y, target_x, target_y,
                                   show_title=False, ax=axis[1, idx_col])

                
for ax, label in zip(axis[-1,:], [f"l = {l}" for l in list_kernel_length_scale]):
    ax.set_xlabel(label, rotation=0, size='large')

for ax in axis.flatten()[:-1]:
    ax.get_legend().remove()
    
axis[0, 0].set_ylabel("CNP", size=18)
axis[1, 0].set_ylabel("LNP", size=18)

fig.tight_layout()

# %%
list_num_context = [3, 8, 12, 16]

fig, axis = plt.subplots(4, len(list_num_context),
                         figsize=(3.5*len(list_num_context), 6),
                         sharex=True,
                         sharey=True)


list_kernel_length_scale = []

for idx_plot, num_context in enumerate(list_num_context):
    kernel_length_scale = np.random.uniform(0.2, 0.8)
    list_kernel_length_scale.append(kernel_length_scale)
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=kernel_length_scale)
    
    
    (context_x, context_y, target_x), target_y = draw_single_example_from_arbitrary_gp(
        kernel_length_scale=kernel_length_scale,
        num_context=num_context,
        num_target=100
    )
    
    # --- CNP ---
    plot_preds_from_single_example(model, context_x, context_y, target_x, target_y,
                                   show_title=False, ax=axis[1, idx_plot])
    
    
    # --- LNP ---
    plot_preds_from_single_example(lnp_model, context_x, context_y, target_x, target_y,
                                   show_title=False, ax=axis[2, idx_plot])
    
    # --- HNPC ---
    plot_preds_from_single_example(hnpc_model, context_x, context_y, target_x, target_y,
                                   show_title=False, ax=axis[3, idx_plot])
    
    # --- GP ---
    gp = tfd.GaussianProcessRegressionModel(
                kernel=kernel,
                index_points=target_x,
                observation_index_points=context_x,
                observations=tf.squeeze(context_y),
                jitter=1.0e-4
    )

    context_x = tf.squeeze(context_x)
    target_x = tf.squeeze(target_x)
    gp_mean_predict = gp.mean()
    gp_std_predict = gp.stddev()
    
    idx_x_sorted = tf.argsort(target_x)
    
    target_x = tf.gather(target_x, idx_x_sorted)
    target_y = tf.gather(target_y, idx_x_sorted)
    gp_mean_predict = tf.gather(gp_mean_predict, idx_x_sorted)
    gp_std_predict = tf.gather(gp_std_predict, idx_x_sorted)

    plot_mean_with_std(x=target_x.numpy(),
                       mean=gp_mean_predict.numpy(),
                       std=gp_std_predict.numpy(),
                       y_true=target_y.numpy(),
                       ax=axis[0, idx_plot]
    )
    
    axis[0, idx_plot].scatter(context_x.numpy(), context_y.numpy())



list_labels = [f"n_context = {n}\nl = {l:.2f}" for n, l in zip(list_num_context, list_kernel_length_scale)]

for ax, label in zip(axis[0, :], list_labels):
    ax.set_title(label, size=18)

for ax in axis.flatten()[:-1]:
    ax.get_legend().remove()
    
axis[0, 0].set_ylabel("GP", size=20)
axis[1, 0].set_ylabel("CNP", size=20)
axis[2, 0].set_ylabel("LNP", size=20)
axis[3, 0].set_ylabel("HNPC", size=20)
    
fig.tight_layout()

# %%



