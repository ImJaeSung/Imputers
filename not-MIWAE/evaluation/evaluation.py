# %%
import pandas as pd
import numpy as np
from collections import namedtuple
from evaluation import metrics_impute, metrics_stat, metrics_MLu, metrics_privacy
from modules import utils

import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")

Metrics = namedtuple(
    "Metrics",
    [   
        "smape",
        "error",
        "arsmape",
        "rmse",
        "armse",
        "mae",
        "armae",
        "KL",
        "GoF",
        "MMD",
        "WD",
        "base_reg", 
        "syn_reg", 
        "base_cls", 
        "syn_cls",
        "model_selection", 
        "feature_selection",
        "Kanon_base",
        "Kanon_syn",
        "KMap",
        "DCR_RS",
        "DCR_RR",
        "DCR_SS",
        "AD",
    ]
)
#%%
def evaluate(imputed, train_dataset, test_dataset, config):
    
    print("\n1. Element-wise Distance with original...")
    smape, error = metrics_impute.elementwise(train_dataset, imputed)
    arsmape = smape + error
    
    print("\n2. Statistical Fidelity: ARMSE...")
    rmse, error = metrics_impute.RootMeanSquaredError(train_dataset, imputed)
    armse = rmse + error
    
    print("\n3. Statistical Fidelity: ARMAE...")
    mae, error = metrics_impute.MeanAbsoluteError(train_dataset, imputed)
    armae = mae + error

    print("\n4. Statistical Fidelity: KL-Divergence...")
    KL = metrics_stat.KLDivergence(train_dataset, imputed)
    
    print("\n5. Statistical Fidelity: Goodness Of Fit...")
    GoF = metrics_stat.GoodnessOfFit(train_dataset, imputed)
    
    print("\n6. Statistical Fidelity: MMD...")
    if config["dataset"] == "covtype":
        MMD = metrics_stat.MaximumMeanDiscrepancy(train_dataset, imputed, large=True)
    else:
        MMD = metrics_stat.MaximumMeanDiscrepancy(train_dataset, imputed)
    
    print("\n7. Statistical Fidelity: Wasserstein...")
    if config["dataset"] == "covtype":
        WD = metrics_stat.WassersteinDistance(train_dataset, imputed, large=True)
    else:
        WD = metrics_stat.WassersteinDistance(train_dataset, imputed)
    
    print("\n8. Machine Learning Utility: Regression...")
    base_reg, syn_reg = metrics_MLu.MLu_reg(train_dataset, test_dataset, imputed)
    
    print("\n9. Machine Learning Utility: Classification...")
    base_cls, syn_cls, model_selection, feature_selection = metrics_MLu.MLu_cls(train_dataset, test_dataset, imputed)
    
    print("\n10. Privacy: K-anonimity...")
    Kanon_base, Kanon_syn = metrics_privacy.kAnonymization(train_dataset, imputed)
    
    print("\n11. Privacy: K-Map...")
    KMap = metrics_privacy.kMap(train_dataset, imputed)
    
    print("\n12. Privacy: DCR...")
    DCR_RS, DCR_RR, DCR_SS = metrics_privacy.DCR_metric(train_dataset, imputed)
    
    print("\n13. Privacy: Attribute Disclosure...")
    AD = metrics_privacy.AttributeDisclosure(train_dataset, imputed)
    
    # print("\n14. Marginal Distribution...")
    # figs = utils.marginal_plot(train_dataset.raw_data, imputed, config)

    return Metrics(
        smape, error, arsmape, rmse, armse, mae, armae, 
        KL, GoF, MMD, WD, 
        base_reg, syn_reg, base_cls, syn_cls, model_selection, feature_selection,
        Kanon_base, Kanon_syn, KMap, DCR_RS, DCR_RR, DCR_SS, AD
    )