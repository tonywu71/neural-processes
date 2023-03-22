#%%
import os
os.chdir("/Users/baker/Documents/MLMI4/conditional-neural-processes/")
from utils.load_model import *
tf.config.set_visible_devices([], 'GPU') # Disable the GPU if present, we wont need it
import matplotlib.pyplot as plt
import numpy as np
# ================================ Training parameters ===============================================

# Regression
args = argparse.Namespace(epochs=60, batch=64, task='regression', num_context=25, uniform_sampling=True, model='HNPC')

pth = f'.data/{args.model}_model_{args.task}_context_{args.num_context}_uniform_sampling_{args.uniform_sampling}/'# + "cp-0030.ckpt"
model, train_ds, test_ds = load_model_and_dataset(args, pth)






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




            
tf.random.set_seed(3)
test_iter = iter(test_ds)
x = next(test_iter)

(context_x, context_y, target_x), target_y = x
pred_y = model(x[0])
fig = plot_regression(target_x, target_y, context_x, context_y, pred_y)
#fig.suptitle(f'loss {logs["loss"]:.5f}')
#%%


