#%%
"""
Reference:
[1] Synthcity: facilitating innovative use cases of synthetic data in different data modalities
- https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/eval_statistical.py
"""
#%%
import numpy as np
from tqdm import tqdm
import torch

from geomloss import SamplesLoss
from scipy.stats import entropy, ks_2samp
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from modules import utils
#%%
def KLDivergence(train_dataset, imputed):
    """
    Marginal statistical fidelity: KL-Divergence
    : lower is better
    """
    
    train = train_dataset.raw_data
    num_bins = 10
    
    # get distribution of continuous variables
    cont_freqs = utils.get_frequency(
        train[train_dataset.continuous_features], 
        imputed[train_dataset.continuous_features], 
        n_histogram_bins=num_bins
    )
    
    result = [] 
    for col in train.columns:
        if col in train_dataset.continuous_features:
            gt, syn = cont_freqs[col]
            kl_div = entropy(syn, gt)
        else:
            pmf_p = train[col].value_counts(normalize=True)
            pmf_q = imputed[col].value_counts(normalize=True)
            
            # Ensure that both PMFs cover the same set of categories
            all_categories = pmf_p.index.union(pmf_q.index)
            pmf_p = pmf_p.reindex(all_categories, fill_value=0)
            pmf_q = pmf_q.reindex(all_categories, fill_value=0)
            
            # Avoid division by zero and log(0) by filtering out zero probabilities
            non_zero_mask = (pmf_p > 0) & (pmf_q > 0)
            
            kl_div = np.sum(pmf_q[non_zero_mask] * np.log(pmf_q[non_zero_mask] / pmf_p[non_zero_mask]))
        result.append(kl_div)
    return np.mean(result)
#%%
def GoodnessOfFit(train_dataset, imputed):
    """
    Marginal statistical fidelity: Kolmogorov-Smirnov test & Chi-Squared test
    : lower is better
    """
    
    train = train_dataset.raw_data
    
    result = [] 
    for col in train.columns:
        if col in train_dataset.continuous_features:
            # Compute the Kolmogorov-Smirnov test for goodness of fit.
            statistic, _ = ks_2samp(train[col], imputed[col])
        else:
            pmf_p = train[col].value_counts(normalize=True) # expected
            pmf_q = imputed[col].value_counts(normalize=True) # observed
            
            # Ensure that both PMFs cover the same set of categories
            all_categories = pmf_p.index.union(pmf_q.index)
            pmf_p = pmf_p.reindex(all_categories, fill_value=0)
            pmf_q = pmf_q.reindex(all_categories, fill_value=0)
            
            # Avoid division by zero and log(0) by filtering out zero probabilities
            non_zero_mask = pmf_p > 0
            
            # Compute the Chi-Squared test for goodness of fit.
            statistic = ((pmf_q[non_zero_mask] - pmf_p[non_zero_mask]) ** 2 / pmf_p[non_zero_mask]).sum()
        result.append(statistic)
    return np.mean(result)
#%%
def MaximumMeanDiscrepancy(train_dataset, imputed, large=False):
    """
    Joint statistical fidelity: Maximum Mean Discrepancy (MMD)
    : lower is better
    
    - MMD using RBF (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    """
    
    train = train_dataset.raw_data
    # only continuous
    train = train[train_dataset.continuous_features]
    imputed = imputed[train_dataset.continuous_features]
    
    if large:
        train = train.sample(frac=0.05, random_state=0)
        imputed = imputed.sample(frac=0.05, random_state=0)
    
    scaler = StandardScaler().fit(train)
    train_ = scaler.transform(train)
    imputed_ = scaler.transform(imputed)
    
    XX = metrics.pairwise.rbf_kernel(
        train_.reshape(len(train_), -1),
        train_.reshape(len(train_), -1),
    )
    YY = metrics.pairwise.rbf_kernel(
        imputed_.reshape(len(imputed_), -1),
        imputed_.reshape(len(imputed_), -1),
    )
    XY = metrics.pairwise.rbf_kernel(
        train_.reshape(len(train_), -1),
        imputed_.reshape(len(imputed_), -1),
    )
    MMD = XX.mean() + YY.mean() - 2 * XY.mean()
    return MMD
#%%
def WassersteinDistance(train_dataset, imputed, large=False):
    """
    Joint statistical fidelity: Wasserstein Distance
    : lower is better
    """
    
    train = train_dataset.raw_data
    # only continuous
    train = train[train_dataset.continuous_features]
    imputed = imputed[train_dataset.continuous_features]
    
    if large:
        train = train.sample(frac=0.05, random_state=0)
        imputed = imputed.sample(frac=0.05, random_state=0)
    
    train_ = train.values.reshape(len(train), -1)
    imputed_ = imputed.values.reshape(len(imputed), -1)
    
    # assert len(train_) == len(imputed_)

    scaler = StandardScaler().fit(train_)
    train_ = scaler.transform(train_).astype(np.float32)
    imputed_ = scaler.transform(imputed_).astype(np.float32)
    
    train_ = torch.tensor(train_, dtype=torch.float32)
    imputed_ = torch.tensor(imputed_, dtype=torch.float32)

    OT_solver = SamplesLoss(loss="sinkhorn")
    """
    Compute WD for 4000 samples and average due to the following error:
    "NameError: name 'generic_logsumexp' is not defined"
    """
    if len(train_) > 4000:
        WD = []
        iter_ = len(train_) // 4000 + 1
        for _ in tqdm(range(iter_), desc="Batch WD..."):
            idx = np.random.choice(range(len(train_)), 4000, replace=False)
            WD.append(OT_solver(train_[idx, :], imputed_[idx, :]).cpu().numpy().item())
        WD = np.mean(WD)
    else:
        WD = OT_solver(train_, imputed_).cpu().numpy().item()
    return WD
#%%
def phi(s, D):
    return (1 + (4 * s) / (2 * D - 3)) ** (-1 / 2)
#%%
def CramerWoldDistance(train_dataset, imputed, config, device):
    """
    Joint statistical fidelity: Cramer-Wold Distance
    : lower is better
    """
    
    train = train_dataset.raw_data
    # only continuous
    train = train[train_dataset.continuous_features]
    imputed = imputed[train_dataset.continuous_features]
    if config["dataset"] == "adult": ### OOM
        train = train.sample(frac=0.5, random_state=42)
        imputed = imputed.sample(frac=0.5, random_state=42)
    
    scaler = StandardScaler().fit(train)
    train_ = scaler.transform(train).astype(np.float32)
    imputed_ = scaler.transform(imputed).astype(np.float32)
    train_ = torch.from_numpy(train_).to(device)
    imputed_ = torch.from_numpy(imputed_).to(device)
    
    gamma_ = (4 / (3 * train_.size(0))) ** (2 / 5)
    
    cw1 = torch.cdist(train_, train_) ** 2 
    cw2 = torch.cdist(imputed_, imputed_) ** 2 
    cw3 = torch.cdist(train_, imputed_) ** 2 
    cw = phi(cw1 / (4 * gamma_), D=train_.size(1)).sum()
    cw += phi(cw2 / (4 * gamma_), D=train_.size(1)).sum()
    cw += -2 * phi(cw3 / (4 * gamma_), D=train_.size(1)).sum()
    cw /= (2 * train_.size(0) ** 2 * torch.tensor(torch.pi * gamma_).sqrt())
    return cw.cpu().numpy().item()