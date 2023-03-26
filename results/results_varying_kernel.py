#%%
import os
#os.chdir("/Users/baker/Documents/MLMI4/conditional-neural-processes/")
os.chdir('/Users/wnbak/Documents/conditional-neural-processes/')
from utils.load_model import *
tf.config.set_visible_devices([], 'GPU') # Disable the GPU if present, we wont need it
from utils.gaussian_processes.gp_model import plot_mean_with_std
from utils.gaussian_processes.plot_gp_utils import plot_preds_from_ds_test
import matplotlib.pyplot as plt
from utils.gaussian_processes.plot_gp_utils import plot_preds_from_single_example
from dataloader.dataloader_for_plotting.load_regression_data_from_arbitrary_gp_varying_kernel import draw_single_example_from_arbitrary_gp
#

args = argparse.Namespace(epochs=60, batch=1024, task='regression_varying', num_context=25, uniform_sampling=True, model='CNP')
model, train_ds, test_ds = load_model_and_dataset(args, ".data/CNP_model_regression_varying_context_25_uniform_sampling_True/")
args = argparse.Namespace(epochs=60, batch=1024, task='regression_varying', num_context=25, uniform_sampling=True, model='LNP')
lnp_model, _, _ = load_model_and_dataset(args, ".data/LNP_model_regression_varying_context_25_uniform_sampling_True/")
args = argparse.Namespace(epochs=60, batch=1024, task='regression_varying', num_context=25, uniform_sampling=True, model='HNPC')
hnpc_model, _, _ = load_model_and_dataset(args, ".data/HNPC_model_regression_varying_context_25_uniform_sampling_True/")


# %%
list_num_context = [3, 8, 12, 16]

fig, axis = plt.subplots(4, len(list_num_context),
                         figsize=(3.5*len(list_num_context), 6),
                         sharex=True,
                         sharey=True)

MIN_KERNEL_LENGTH_SCALE = 0.1
MAX_KERNEL_LENGTH_SCALE = 1.0
AVG_KERNEL_LENGTH_SCALE = (MIN_KERNEL_LENGTH_SCALE + MAX_KERNEL_LENGTH_SCALE) / 2.0
AVG_KERNEL = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=AVG_KERNEL_LENGTH_SCALE)

list_kernel_length_scales = []


for idx_plot, num_context in enumerate(list_num_context):
    
    
    (context_x, context_y, target_x), target_y, l1, l2 = draw_single_example_from_arbitrary_gp(
        min_kernel_length_scale=MIN_KERNEL_LENGTH_SCALE,
        max_kernel_length_scale=MAX_KERNEL_LENGTH_SCALE,
        num_context=num_context,
        num_target=100
    )
    
    list_kernel_length_scales.append((l1, l2))
    
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
                kernel=AVG_KERNEL,
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



list_labels = [f"n_context = {n}\nl1 = {l1:.2f}, l2 = {l2:.2f}" for n, (l1, l2) in zip(list_num_context, list_kernel_length_scales)]

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



