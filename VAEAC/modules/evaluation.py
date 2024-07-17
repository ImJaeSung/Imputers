# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import namedtuple
from dython.nominal import associations
from modules import metric

import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")

Metrics = namedtuple(
    "Metrics",
    [
        "KS",
        "W1",
        "PCD",
        "DCR_RS",
        "DCR_R",
        "DCR_S",
        "logcluster",
        "MAPE_baseline",
        "MAPE",
        "F1_baseline",
        "F1",
        "AD_F1_1",
        "AD_F1_10",
        "AD_F1_100",
    ],
)
#%%
def evaluate(syndata, train_dataset, test_dataset, config):
    train = train_dataset.raw_data
    test = test_dataset.raw_data
    
    syndata_ = syndata.copy()
    test_1 = test.copy()
    syn_mean = syndata[train_dataset.continuous_features].mean(axis=0)
    syn_std = syndata[train_dataset.continuous_features].std(axis=0)
    syndata_[train_dataset.continuous_features] = (syndata[train_dataset.continuous_features] - syn_mean) / syn_std # scaled
    test_1[train_dataset.continuous_features] = (test[train_dataset.continuous_features] - syn_mean) / syn_std # scaled
    
    train_ = train.copy()
    test_2 = test.copy()
    mean = train[train_dataset.continuous_features].mean(axis=0)
    std = train[train_dataset.continuous_features].std(axis=0)
    train_[train_dataset.continuous_features] = (train[train_dataset.continuous_features] - mean) / std # scaled
    test_2[train_dataset.continuous_features] = (test[train_dataset.continuous_features] - mean) / std # scaled
    
    print("1. Kolmogorov-Smirnov test...")
    print("2. 1-Wasserstein distance...")
    Dn, W1 = metric.statistical_similarity(train_.copy(), syndata_.copy())
    
    print("3. Pairwise correlation difference (PCD)...")
    syn_asso = associations(
        syndata, nominal_columns=train_dataset.categorical_features,
        compute_only=True)
    true_asso = associations(
        train, nominal_columns=train_dataset.categorical_features,
        compute_only=True)
    pcd_corr = np.linalg.norm(true_asso["corr"] - syn_asso["corr"])
    
    print("4. Distance to Closest Record (DCR)...")
    DCR = metric.DCR_metric(train_[train_dataset.continuous_features].copy(), syndata_[train_dataset.continuous_features].copy())
    
    print("5. log-cluster...")
    k = 20
    kmeans = KMeans(n_clusters=k, random_state=config["seed"])
    kmeans.fit(pd.concat([train_[train_dataset.continuous_features], syndata_[train_dataset.continuous_features]], axis=0))
    logcluster = 0
    for c in range(k):
        n_total = (kmeans.labels_ == c).sum()
        n_train = (kmeans.labels_[: len(train)] == c).sum()
        logcluster += (n_train / n_total - 0.5) ** 2
    logcluster /= k
    logcluster = np.log(logcluster)
    
    print("6. Machine Learning Utility in Regression (MAPE)...")
    base_reg = []
    syn_reg = []
    for target in train_dataset.continuous_features:
        target_ = train_[target] * std[target] + mean[target]
        if (target_ == 0).sum() > 0: continue # zero ratio
        if target == "OWN_CAR_AGE": continue # unstable results
        if target == "DAYS_EMPLOYED": continue # unstable results
        if target == "Mortgage": continue # unstable results
        if target == "CCAvg": continue # unstable results
        if target == "Hillshade_9am": continue # unstable results
        if target == "Hillshade_Noon": continue # unstable results
        if target == "Horizontal_Distance_To_Fire_Points": continue # unstable results
        if target == "Blast Furnace Slag": continue # unstable results
        if target == "Fly Ash": continue # unstable results
        if target == "Superplasticizer": continue # unstable results
        if target == "concavity1": continue # unstable results
        if target == "concavity2": continue # unstable results
        if target == "concave points1": continue # unstable results
        if target == "pox": continue # unstable results
        if target == "vac": continue # unstable results
        if target == "nuc": continue # unstable results
        if config["dataset"] == "banknote": continue # unstable results
        if config["dataset"] == "letter": continue # unstable results
        if config["dataset"] == "spam": continue # unstable results
        if config["dataset"] == "banknote": continue # unstable results
        if config["dataset"] == "yeast": continue # unstable results
        print(f"Target: {target}")
        base_reg.extend(metric.regression_eval(train_.copy(), test_2.copy(), target, mean, std))
        syn_reg.extend(metric.regression_eval(syndata_.copy(), test_1.copy(), target, syn_mean, syn_std))
        
    print("7. Machine Learning Utility in Classification (F1)...")
    base_clf = []
    syn_clf = []
    for target in train_dataset.categorical_features:
        print(f"Target: {target}")
        base_clf.extend(metric.classification_eval(train_.copy(), test_2.copy(), target))
        syn_clf.extend(metric.classification_eval(syndata_.copy(), test_1.copy(), target))
        
    # print("7. Nearest Neighbor Adversarial Accuracy...")
    # AA_train, AA_test, AA = privacyloss(train, test, syndata, data_percent=15)
    
    print("8. Attribute Disclosure...")
    compromised_idx = np.random.choice(
        range(len(train_)), 
        int(len(train_) * 0.01), 
        replace=False)
    compromised = train_.iloc[compromised_idx].reset_index().drop(columns=['index'])
    
    # for attr_num in [1, 2, 3, 4, 5]:
    #     if attr_num > len(dataset.continuous_features): break
    attr_num = 5
    attr_compromised = train_dataset.continuous_features[:attr_num]
    AD_f1 = []
    for K in [1, 10, 100]:
        acc, f1 = metric.attribute_disclosure(
            K, compromised, syndata_.copy(), attr_compromised, train_dataset)
        AD_f1.append(f1)
        
    print("9. Marginal Distribution...")
    figs = metric.marginal_plot(train, syndata, config)

    return Metrics(
        np.mean(Dn),
        np.mean(W1),
        pcd_corr,
        DCR[0],
        DCR[1],
        DCR[2],
        logcluster,
        np.mean([x[1] for x in base_reg]),
        np.mean([x[1] for x in syn_reg]),
        np.mean([x[1] for x in base_clf]),
        np.mean([x[1] for x in syn_clf]),
        AD_f1[0],
        AD_f1[1],
        AD_f1[2],
    ), figs
#%%