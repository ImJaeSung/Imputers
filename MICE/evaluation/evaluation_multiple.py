# %%
from tqdm import tqdm
import importlib

import pandas as pd
import numpy as np

from collections import namedtuple
from modules.utils import undummify
#%%
Metrics = namedtuple(
    "Metrics",
    [
        "bias", "bias_percent", "coverage", "interval"
    ],
)

def evaluate(train_dataset, config, M=100):
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    
    """target estimand"""
    data = train_dataset.raw_data[train_dataset.continuous_features]
    true = (data > data.mean(axis=0)).astype(float).mean(axis=0)

    est = []
    var = []

    # prepare the store for the imputations
    for s in tqdm(range(M),desc='Multiple_imputations...'):
        
        model = model_module.MICE(
            max_iter=config["max_iter"],
            random_state=s
        )

        imputed_data = model.fit_transform(np.array(train_dataset.data))
        imputed = pd.DataFrame(imputed_data, columns=train_dataset.data.columns)
        imputed = undummify(imputed, prefix_sep='###')

        # un-standardization of synthetic data
        for col, scaler in train_dataset.scalers.items():
            imputed[[col]] = scaler.inverse_transform(imputed[[col]])

        # post-process
        imputed[train_dataset.integer_features] = imputed[train_dataset.integer_features].round(0).astype(int)

        data = imputed[train_dataset.continuous_features]
        binary = (data > data.mean(axis=0)).astype(float)
        p = binary.mean(axis=0)
        est.append(p)
        var.append(p * (1. - p) / len(binary))
        
    Q = np.mean(est, axis=0)
    U = np.mean(var, axis=0) + (M + 1) / M * np.var(est, axis=0, ddof=1)
    lower = Q - 1.96 * np.sqrt(U)
    upper = Q + 1.96 * np.sqrt(U)
    
    bias = float(np.abs(Q - true).mean())
    bias_percent = 100 * float((np.abs((Q - true)/Q)).mean())
    coverage = float(((lower < true) & (true < upper)).mean())
    interval = float((upper - lower).mean())
    
    return Metrics(
        bias, bias_percent, coverage, interval
    )