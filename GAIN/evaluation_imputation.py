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
# from VAEAC.no_datasets import compute_normalization

from utils import  normalization, renormalization, rounding,test_loss, generator_loss, discriminator_loss, sample_M,sample_Z


import warnings
warnings.filterwarnings("ignore")

Metrics = namedtuple(
    "Metrics",
    [
        "bias", "coverage", "interval"
    ],
)

def evaluate(train_dataset,total_dataloader,parameters, G,D, M=100):
    """target estimand"""
    data = train_dataset.raw_data[train_dataset.continuous_features]
    true = (data > data.mean(axis=0)).astype(float).mean(axis=0)
    

    total_column = train_dataset.continuous_features + train_dataset.categorical_features


    est = []
    var = []
    #full_imputed = model.impute(train_dataset, L=200, M=M, seed=0)
    use_cuda = torch.cuda.is_available()

    # prepare the store for the imputations
    for s in tqdm(range(M),desc='Multiple_imputations'):
        results = []

        imputed_data_list = []
        for i, inference_data in enumerate(total_dataloader):
            X_inf = inference_data.float()
            get_batch = X_inf.shape[0]
            dim = X_inf.shape[1]

            M_inf = 1 - torch.isnan(X_inf).float() ## 1 : observed 0 : nan
            X_inf = torch.nan_to_num(X_inf,nan=0.0)     
            Z_inf = sample_Z(get_batch, dim).float()
            New_X_inf = M_inf * X_inf + (1-M_inf) * Z_inf  # Missing Data Introduce
            MSE_final, Sample = test_loss(G,X_inf, M_inf, New_X_inf)

            imputed_data = M_inf * X_inf + (1-M_inf) * Sample

            imputed_data_list.append(imputed_data)
                
        all_imputed_data = torch.cat(imputed_data_list,dim=0)


        final_data = renormalization(np.array(all_imputed_data.detach()) ,parameters)
        real_final_data = rounding(final_data,np.array(train_dataset.raw_data))
        imputed = pd.DataFrame(real_final_data, columns = total_column)


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
