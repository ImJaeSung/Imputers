# %%
import numpy as np
from collections import namedtuple
from tqdm import tqdm
#%%
Metrics = namedtuple(
    "Metrics",
    [
        "bias", "bias_percent", "coverage", "interval"
    ],
)
#%%
def evaluate(train_dataset, G, config, device, M=50):
    """target estimand"""
    data = train_dataset.raw_data[train_dataset.continuous_features]
    true = (data > data.mean(axis=0)).astype(float).mean(axis=0)
    
    est = []
    var = []

    # prepare the store for the imputations
    for s in tqdm(range(M), desc='Multiple imputations...'):
        imputed = G.impute(train_dataset, config, device)
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
