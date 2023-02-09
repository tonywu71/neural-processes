from typing import Callable, Optional, Tuple
import numpy as np
import pandas as pd


def get_df_from_1d_function(fct: Callable, x_arr: np.ndarray) -> pd.DataFrame:
    fct = np.vectorize(fct)
    df = pd.DataFrame({"x": x_arr, "y": fct(x_arr)})
    return df


def split_train_test(df: pd.DataFrame, frac: float, random_state: Optional[int]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df.sample(frac=frac, random_state=random_state)
    df_test = df.drop(df_train.index)
    return df_train, df_test
