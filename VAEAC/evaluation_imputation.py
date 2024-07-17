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
from modules.imputation_networks import get_imputation_networks
from modules.train_utils import extend_batch, get_validation_iwae
import warnings
warnings.filterwarnings("ignore")

Metrics = namedtuple(
    "Metrics",
    [
        "bias", "coverage", "interval"
    ],
)
#%%

def compute_normalization(data, one_hot_max_sizes):
    """
    Compute the normalization parameters (i. e. mean to subtract and std
    to divide by) for each feature of the dataset.
    For categorical features mean is zero and std is one.
    i-th feature is denoted to be categorical if one_hot_max_sizes[i] >= 2.
    Returns two vectors: means and stds.
    """
    norm_vector_mean = torch.zeros(len(one_hot_max_sizes))
    norm_vector_std = torch.ones(len(one_hot_max_sizes))
    for i, size in enumerate(one_hot_max_sizes):
        if size >= 2:
            continue
        v = data[:, i]
        v = v[~torch.isnan(v)]
        vmin, vmax = v.min(), v.max()
        vmean = v.mean()
        vstd = v.std()
        norm_vector_mean[i] = vmean
        norm_vector_std[i] = vstd
    return norm_vector_mean, norm_vector_std 



def evaluate(train_dataset, model,networks,one_hot_max_sizes, M=10):
    """target estimand"""
    data = train_dataset.raw_data[train_dataset.continuous_features]
    true = (data > data.mean(axis=0)).astype(float).mean(axis=0)
    


    total_column = train_dataset.continuous_features + train_dataset.categorical_features

    raw_data = np.array(train_dataset.data)
    raw_data = torch.from_numpy(raw_data).float()
    norm_mean, norm_std = compute_normalization(raw_data, one_hot_max_sizes)
    norm_std = torch.max(norm_std, torch.tensor(1e-9))


    data = (raw_data - norm_mean[None]) / norm_std[None]

    batch_size = 64

    est = []
    var = []
    #full_imputed = model.impute(train_dataset, L=200, M=M, seed=0)
    use_cuda = torch.cuda.is_available()
    dataloader = DataLoader(data, batch_size=batch_size,
                    shuffle=False, num_workers=0,
                    drop_last=False)
    #%%
    # prepare the store for the imputations


    iterator = dataloader

    # ========================데이터 생성==========================
    for s in tqdm(range(M),desc='Multiple_imputations'):
        results = []
        for batch in iterator:


            batch_extended = torch.tensor(batch)
            batch_extended = extend_batch(batch_extended, dataloader, batch_size)

            if use_cuda:
                batch = batch.cuda()
                batch_extended = batch_extended.cuda()

            mask_extended = torch.isnan(batch_extended).float()

            with torch.no_grad():
                samples_params = model.generate_samples_params(batch_extended,
                                                            mask_extended,
                                                            1)
                samples_params = samples_params[:batch.shape[0]]

            mask = torch.isnan(batch)
            batch_zeroed_nans = torch.tensor(batch)
            batch_zeroed_nans[mask] = 0

            # impute samples from the generative distributions into the data
            # and save it to the results
            
            sample_params = samples_params[:, 0]
            sample = networks['sampler'](sample_params)
            sample[torch.logical_not(mask)] = 0 
            sample += batch_zeroed_nans
            results.append(torch.tensor(sample, device='cpu'))
        

        results = torch.cat(results).unsqueeze(1)
        result = results.view(results.shape[0] , results.shape[2])


        result = result * norm_std[None] + norm_mean[None]

        imputed = pd.DataFrame(result.numpy(), columns = total_column)

        data = imputed[train_dataset.continuous_features]
        binary = (data > data.mean(axis=0)).astype(float)
        p = binary.mean(axis=0)
        est.append(p)
        var.append(p * (1. - p) / len(binary))

#%%

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
#%%