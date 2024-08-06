#%%
import numpy as np
# %%
"""
https://github.com/vanderschaarlab/hyperimpute/blob/main/src/hyperimpute/plugins/utils/metrics.py
"""
def MeanAbsoluteError(train_dataset, imputed):
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

    mean = np.mean(original, axis=0)
    std = np.std(original, axis=0)

    original_ = (original - mean) / std
    imputation_ = (imputation - mean) / std

    original_ = original_[train_dataset.mask[:, :C] == 1]
    imputation_ = imputation_[train_dataset.mask[:, :C] == 1]

    mae = np.absolute(original_ - imputation_).sum() 
    mae /= train_dataset.mask[:, :C].sum()

    """categorical"""
    original = train_dataset.raw_data.values[:, C:]
    original = original[train_dataset.mask[:, C:] == 1]
   
    imputation = imputed.values[:, C:]
    imputation = imputation[train_dataset.mask[:, C:] == 1]
    
    error = 1. - (original == imputation).mean()
    
    return mae, error
#%%
def RootMeanSquaredError(train_dataset, imputed):
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

    mean = np.mean(original, axis=0)
    std = np.std(original, axis=0)

    original_ = (original - mean) / std
    imputation_ = (imputation - mean) / std

    original_ = original_[train_dataset.mask[:, :C] == 1]
    imputation_ = imputation_[train_dataset.mask[:, :C] == 1]
    
    rmse = ((original_ - imputation_) ** 2).sum()
    rmse /= train_dataset.mask[:, :C].sum()
    rmse = np.sqrt(rmse)

    """categorical"""
    original = train_dataset.raw_data.values[:, C:]
    original = original[train_dataset.mask[:, C:] == 1]
   
    imputation = imputed.values[:, C:]
    imputation = imputation[train_dataset.mask[:, C:] == 1]
    
    error = 1. - (original == imputation).mean()

    return rmse, error
#%%
def elementwise(train_dataset, imputed):
    """continuous"""
    C = train_dataset.num_continuous_features
    original = train_dataset.raw_data.values[:, :C]
    original = original[train_dataset.mask[:, :C] == 1]
    
    imputation = imputed.values[:, :C]
    imputation = imputation[train_dataset.mask[:, :C] == 1]
    
    smape = np.abs(original - imputation).astype(np.float32)
    smape /= (np.abs(original) + np.abs(imputation)) + 1e-6 # numerical stability
    smape = smape.mean()
    
    """categorical"""
    original = train_dataset.raw_data.values[:, C:]
    original = original[train_dataset.mask[:, C:] == 1]
    
    imputation = imputed.values[:, C:]
    imputation = imputation[train_dataset.mask[:, C:] == 1]
    
    error = 1. - (original == imputation).mean()
    
    return smape, error
#%%