# %%
import pandas as pd
import numpy as np
from collections import namedtuple
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
# from VAEAC.no_datasets import compute_normalization
#from modules.imputation_networks import get_imputation_networks
#from modules.train_utils import extend_batch, get_validation_iwae
import warnings
warnings.filterwarnings("ignore")

#%%
Metrics = namedtuple(
    "Metrics",
    [
        "bias", "coverage", "interval"
    ],
)

def evaluate(train_dataset,seed, M=10,):
    
    """target estimand"""
    data = train_dataset.raw_data[train_dataset.continuous_features]
    true = (data > data.mean(axis=0)).astype(float).mean(axis=0)
    total_column = train_dataset.continuous_features + train_dataset.categorical_features
    raw_data = np.array(train_dataset.data)

    est = []
    var = []

    # prepare the store for the imputations

    for s in tqdm(range(M),desc='Multiple_imputations'):
        results = []
        iterative_num = seed+20

        imputer = IterativeImputer(
            imputation_order='ascending',  
            random_state=s,  
            max_iter=iterative_num
        ) 

        imputed_data = imputer.fit_transform(raw_data)
        imputed = pd.DataFrame(imputed_data, columns = total_column)

        data = imputed[train_dataset.continuous_features]
        binary = (data > data.mean(axis=0)).astype(float)
        p = binary.mean(axis=0)
        est.append(p)
        var.append(p * (1. - p) / len(binary))

    # concatenate all batches into one [n x K x D] tensor,
    # where n in the number of objects, K is the number of imputations
    # and D is the dimensionality of one object


        
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