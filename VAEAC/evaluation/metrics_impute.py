#%%
import numpy as np
# %%
import numpy as np
# %%
"""
Reference:
[1] https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/utils/metrics.py
"""
def NMAE(train_dataset, imputed):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth.

    Args
    ----------
        train_dataset : module including Ground truth data and mask.
        imputed : Data with imputed variables.

    Returns
    ----------
        mae, accaracy

    """
    assert np.isnan(imputed).sum().sum() == 0
    
    C = train_dataset.num_continuous_features
    
    """continuous"""
    original = train_dataset.raw_data.values[:, :C]
    imputation = imputed.values[:, :C]
    
    original_ = original[train_dataset.mask[:, :C] == 1]
    imputation_ = imputation[train_dataset.mask[:, :C] == 1]

    mae = np.mean(np.absolute(original_ - imputation_), axis=0)
    std = np.std(original_, axis=0)
    
    nmae = mae / std

    """categorical"""
    original = train_dataset.raw_data.values[:, C:]
    original = original[train_dataset.mask[:, C:] == 1]
   
    imputation = imputed.values[:, C:]
    imputation = imputation[train_dataset.mask[:, C:] == 1]
    
    error = 1. - (original == imputation).mean()

    return nmae, error
#%%
def NRMSE(train_dataset, imputed):
    """
    Root Mean Squared Error (RMSE) between imputed variables and ground truth

    Args
    ----------
        train_dataset : module including Ground truth data and mask.
        imputed : Data with imputed variables.

    Returns
    ----------
        rmse, accaracy

    """
    assert np.isnan(imputed).sum().sum() == 0

    C = train_dataset.num_continuous_features
    
    """continuous"""
    original = train_dataset.raw_data.values[:, :C]
    imputation = imputed.values[:, :C]
    
    original_ = original[train_dataset.mask[:, :C] == 1]
    imputation_ = imputation[train_dataset.mask[:, :C] == 1]

    rmse = np.mean(((original_ - imputation_) ** 2), axis=0)
    std = np.std(original_, axis=0)
    
    nrmse = np.sqrt(rmse / std**2)

    """categorical"""
    original = train_dataset.raw_data.values[:, C:]
    original = original[train_dataset.mask[:, C:] == 1]
   
    imputation = imputed.values[:, C:]
    imputation = imputation[train_dataset.mask[:, C:] == 1]
    
    error = 1. - (original == imputation).mean()

    return nrmse, error
#%%
def elementwise(train_dataset, imputed):
    """continuous"""
    C = train_dataset.num_continuous_features
    original = train_dataset.raw_data.values[:, :C]
    original = original[train_dataset.mask[:, :C] == 1]
    
    imputation = imputed.values[:, :C]
    imputation = imputation[train_dataset.mask[:, :C] == 1]
    
    smape = np.abs(original - imputation)
    smape /= (np.abs(original) + np.abs(imputation)) + 1e-6 # numerical stability
    smape = smape.mean()
    
    """categorical"""
    original = train_dataset.raw_data.values[:, C:]
    original = original[train_dataset.mask[:, C:] == 1]
    
    imputation = imputed.values[:, C:]
    imputation = imputation[train_dataset.mask[:, C:] == 1]
    
    # accuracy = (original == imputation).mean()
    error = 1. - (original == imputation).mean()
    
    return smape, error
#%%