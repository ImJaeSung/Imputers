# %%
import pandas as pd
import numpy as np
from collections import namedtuple
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

Metrics = namedtuple(
    "Metrics",
    [
        "bias", "coverage", "interval"
    ],
)
#%%
def evaluate(train_dataset, full_imputed, M=100):
    """target estimand"""
    data = train_dataset.raw_data[train_dataset.continuous_features]
    true = (data > data.mean(axis=0)).astype(float).mean(axis=0)

    est = []
    var = []
    for m in tqdm(range(M), desc="Multiple Imputaion"):
        imputed = pd.DataFrame(full_imputed[:,:, m], columns=train_dataset.features)
        
        assert imputed.shape == train_dataset.raw_data.shape
        
        data = imputed[train_dataset.continuous_features]
        binary = (data > data.mean(axis=0)).astype(float)
        
        p = binary.mean(axis=0)
        est.append(p)
        var.append(p * (1. - p) / len(binary))
        # est.append(
        #     pd.DataFrame(imputed[train_dataset.continuous_features].mean(axis=0))
        # )
        # var.append(
        #     pd.DataFrame(imputed[train_dataset.continuous_features].var(axis=0, ddof=1) / len(imputed))
        # )
        
    Q = np.mean(est, axis=0)
    U = np.mean(var, axis=0) + (M + 1) / M * np.var(est, axis=0, ddof=1)
    lower = Q - 1.96 * np.sqrt(U)
    upper = Q + 1.96 * np.sqrt(U)
    
    bias = float(np.abs(Q - true).mean())
    coverage = float(((lower < true) & (true < upper)).mean())
    interval = float((upper - lower).mean())
    
    return Metrics(
        bias, coverage, interval
    )
#%%