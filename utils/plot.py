from typing import Optional
import pandas as pd
import tensorflow as tf


def plot_learning_curve(history: tf.keras.callbacks.History, filepath: Optional[str]=None):
    history_loss = pd.DataFrame(history.history, columns=["loss", "val_loss"])
    ax = history_loss.plot(xlabel="Epochs", ylabel="Validation loss (cross-entropy)",
                           title="Learning Curve")

    if filepath is not None:
        fig = ax.get_figure() # type: ignore
        fig.savefig(filepath)
    
    return
