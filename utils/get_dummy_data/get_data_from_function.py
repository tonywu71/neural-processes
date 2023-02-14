from typing import Callable, Optional, Tuple
import numpy as np
import pandas as pd


def get_df_from_1d_function(fct: Callable, x_arr: np.ndarray) -> pd.DataFrame:
    """Returns a DataFrame with columns `x` and `y` from a function and
    an array of x values."""
    fct = np.vectorize(fct)
    df = pd.DataFrame({"x": x_arr, "y": fct(x_arr)})
    return df


def split_train_test(df: pd.DataFrame, frac: float,
                     random_state: Optional[int]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns a tuple of DataFrames with the first one being the training set
    and the second one the test set."""""
    df_train = df.sample(frac=frac, random_state=random_state)
    df_test = df.drop(df_train.index)
    return df_train, df_test
