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
        "CW",
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
def evaluate(imputed, train_dataset, test_dataset, config, device):
    
    print("\n1. Element-wise: ARSMAPE...")
    smape, error = metrics_impute.SMAPE(train_dataset, imputed)
    arsmape = smape + error
    
    print("\n2. Element-wise: ARMSE...")
    rmse, error = metrics_impute.NRMSE(train_dataset, imputed)
    armse = rmse + error
    
    print("\n3. Element-wise: ARMAE...")
    mae, error = metrics_impute.NMAE(train_dataset, imputed)
    armae = mae + error

    print("\n4. Statistical Fidelity: KL-Divergence...")
    KL = metrics_stat.KLDivergence(train_dataset, imputed)
    
    print("\n5. Statistical Fidelity: Goodness Of Fit...")
    GoF = metrics_stat.GoodnessOfFit(train_dataset, imputed)
    
    print("\n6. Statistical Fidelity: MMD...")
    MMD = metrics_stat.MaximumMeanDiscrepancy(train_dataset, imputed)
    
    print("\n7. Statistical Fidelity: Wasserstein...")
    WD = metrics_stat.WassersteinDistance(train_dataset, imputed, device)
    
    print("\n8. Statistical Fidelity: Cramer-Wold Distance...")
    CW = metrics_stat.CramerWoldDistance(train_dataset, imputed, config, device)

    print("\n9. Machine Learning Utility: Regression...")
    base_reg, syn_reg = metrics_MLu.MLu_reg(train_dataset, test_dataset, imputed)
    
    print("\n10. Machine Learning Utility: Classification...")
    base_cls, syn_cls, model_selection, feature_selection = metrics_MLu.MLu_cls(train_dataset, test_dataset, imputed)
    
    print("\n11. Privacy: K-anonimity...")
    Kanon_base, Kanon_syn = metrics_privacy.kAnonymization(train_dataset, imputed)
    
    print("\n12. Privacy: K-Map...")
    KMap = metrics_privacy.kMap(train_dataset, imputed)
    
    print("\n13. Privacy: DCR...")
    DCR_RS, DCR_RR, DCR_SS = metrics_privacy.DCR_metric(train_dataset, imputed)
    
    print("\n14. Privacy: Attribute Disclosure...")
    AD = metrics_privacy.AttributeDisclosure(train_dataset, imputed)
    
    # print("\n14. Marginal Distribution...")
    # figs = utils.marginal_plot(train_dataset.raw_data, imputed, config)

    return Metrics(
        smape, error, arsmape, rmse, armse, mae, armae, 
        KL, GoF, MMD, WD, CW,
        base_reg, syn_reg, base_cls, syn_cls, model_selection, feature_selection,
        Kanon_base, Kanon_syn, KMap, DCR_RS, DCR_RR, DCR_SS, AD
    )