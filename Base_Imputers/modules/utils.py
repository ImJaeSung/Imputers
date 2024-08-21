#%%
import typing as tp
from typing import Any
from tqdm import tqdm

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
import os

#%%
"""for reproducibility"""
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대한 시드 고정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NumPy 시드 고정
    np.random.seed(seed)
    random.seed(seed)   

#%%
def to_ndarray(X: Any) -> np.ndarray:
    """Helper for casting arguments to `numpy.ndarray`.

    Args:
        X: the object to cast.

    Returns:
        pd.DataFrame: the converted ndarray.

    Raises:
        ValueError: if the argument cannot be converted to a ndarray.
    """
    if isinstance(X, np.ndarray):
        return X
    elif isinstance(X, (list, pd.DataFrame, pd.core.series.Series)):
        return np.array(X)

    raise ValueError(
        f"unsupported data type {type(X)}. Try list, pandas.DataFrame or numpy.ndarray"
    )
#%%
def array_to_dataframe(
    data: tp.Union[pd.DataFrame, np.ndarray], columns=None
) -> pd.DataFrame:
    """
    Args:
        data: Pandas DataFrame or Numpy Array
        columns: If data is a Numpy Array, columns needs to be a list of all column names

    Returns:
        Pandas DataFrame with the given data
    """
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(data, np.ndarray), "Input needs to be a Pandas DataFrame or a Numpy Array"
    
    assert (columns), "To convert the data into a Pandas DataFrame, a list of column names has to be given!"
    
    assert len(columns) == len(data[0]), "%d column names are given, but array has %d columns!" % (
        len(columns),
        len(data[0]),
    )

    return pd.DataFrame(data=data, columns=columns)
#%%
def marginal_plot(train_dataset, imputed, config):
    train = train_dataset.raw_data
    if not os.path.exists(f"./assets/figs/{config['dataset']}/seed{config['seed']}/"):
        os.makedirs(f"./assets/figs/{config['dataset']}/seed{config['seed']}/")
    
    figs = []
    for idx, feature in tqdm(enumerate(train.columns), desc="Plotting Histograms..."):
        fig = plt.figure(figsize=(7, 4))
        fig, ax = plt.subplots(1, 1)
        sns.histplot(
            data=imputed,
            x=imputed[feature],
            stat='density',
            label='synthetic',
            ax=ax,
            bins=int(np.sqrt(len(imputed)))) 
        sns.histplot(
            data=train,
            x=train[feature],
            stat='density',
            label='train',
            ax=ax,
            bins=int(np.sqrt(len(train)))) 
        ax.legend()
        ax.set_title(f'{feature}', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"./assets/figs/{config['dataset']}/seed{config['seed']}/hist_{feature}.png")
        # plt.show()
        plt.close()
        figs.append(fig)
    return figs
#%%
def get_frequency(
    X_gt: pd.DataFrame, X_synth: pd.DataFrame, n_histogram_bins: int = 10
):
    """
    Reference:
    [1] https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/_utils.py
    
    Get percentual frequencies for each possible real categorical value.

    Returns:
        The observed and expected frequencies (as a percent).
    """
    res = {}
    for col in X_gt.columns:
        local_bins = min(n_histogram_bins, len(X_gt[col].unique()))

        if len(X_gt[col].unique()) < 5:  # categorical
            gt = (X_gt[col].value_counts() / len(X_gt)).to_dict()
            synth = (X_synth[col].value_counts() / len(X_synth)).to_dict()
        else:
            gt_vals, bins = np.histogram(X_gt[col], bins=local_bins)
            synth_vals, _ = np.histogram(X_synth[col], bins=bins)
            gt = {k: v / (sum(gt_vals) + 1e-8) for k, v in zip(bins, gt_vals)}
            synth = {k: v / (sum(synth_vals) + 1e-8) for k, v in zip(bins, synth_vals)}

        for val in gt:
            if val not in synth or synth[val] == 0:
                synth[val] = 1e-11
        for val in synth:
            if val not in gt or gt[val] == 0:
                gt[val] = 1e-11

        if gt.keys() != synth.keys():
            raise ValueError(f"Invalid features. {gt.keys()}. syn = {synth.keys()}")
        res[col] = (list(gt.values()), list(synth.values()))

    return res

#%%
def undummify(imputed, prefix_sep="###"):
    cols2collapse = {
        col.split(prefix_sep)[0]: (prefix_sep in col) for col in imputed.columns
    }

    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                imputed.filter(like=f"{col}###") # duplication column name
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified.astype(float))
        else:
            series_list.append(imputed[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df 