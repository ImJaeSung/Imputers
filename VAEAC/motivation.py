#%%
import os
from tqdm import tqdm
import argparse
import importlib
import time

import numpy as np
import pandas as pd
from collections import namedtuple


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim

from modules.utils import set_random_seed
#%%
import warnings
warnings.filterwarnings('ignore')
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "dimvae_motivation" # put your WANDB project name
# entity = "wotjd1410" # put your WANDB username

run = wandb.init(
    project=project, 
    # entity=entity, 
    tags=["train"], # put tags of this python project
)
#%%
class ArgParseRange:
    """
    List with this element restricts the argument to be
    in range [start, end].
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __repr__(self):
        return '{0}...{1}'.format(self.start, self.end)
#%%
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')
    
    parser.add_argument('--dataset', type=str, default='nomao', 
                        help="""
                        Dataset options: 
                        loan, kings, banknote, concrete, redwine, 
                        whitewine, breast, letter, abalone, anuran,
                        spam, diabetes, dna, ncbirths
                        """)
    parser.add_argument("--model", default='VAEAC', type=str) 
    parser.add_argument("--seed", default=0, type=int,
                        help="seed for repeatable results") 

    parser.add_argument("--missing_type", default="MAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument("--missing_rate", default=0.3, type=float,
                        help="missing rate")
     
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="the ratio of train test split") 
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')  
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number epochs to train VAEAC.')
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='learning rate to train VAEAC')
    
    parser.add_argument('--validation_ratio', default=0.25, type=float,
                        choices=[ArgParseRange(0, 1)],
                        help='The proportion of objects ' +
                            'to include in the validation set.')

    parser.add_argument('--validation_iwae_num_samples', default=25, 
                        type=int, action='store', 
                        help='Number of samples per object to estimate IWAE ' +
                            'on the validation set. Default: 25.'
                        )

    parser.add_argument('--validations_per_epoch', default=1,
                        type=int, action='store',
                        help='Number of IWAE estimations on the validation set ' +
                            'per one epoch on the training set. Default: 1.'
                        )

    parser.add_argument('--use_last_checkpoint', action='store_true',
                        default=False,
                        help='By default the model with the best ' +
                            'validation IWAE is used to generate ' +
                            'imputations. This flag forces the last model ' +
                            'to be used.'
                        )
    if debug:
        return parser.parse_args(args=[])
    else:
        return parser.parse_args()
    
#%%
EncodedInfo = namedtuple(
    'EncodedInfo', 
    ['num_features', 'num_continuous_features', 'num_categories', 'one_hot_max_sizes'])
#%%
def generate_data(n=1000, seed=42):
    np.random.seed(seed)
    mu = np.zeros(3)
    cov = np.array([
        [1.0, 0.8, 0.6],
        [0.8, 1.0, 0.7],
        [0.6, 0.7, 1.0]
    ])
    data = np.random.multivariate_normal(mu, cov, size=n)
    df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])

    return df, data[:, 0].mean()
#%%
class CustomDataset(Dataset):
    def __init__(self):
        
        data, _ = generate_data()
        
        self.continuous_features = ['x1', 'x2', 'x3']
        self.categorical_features = []
        self.integer_features = []
        self.ClfTarget = []
        
        self.features = self.continuous_features + self.categorical_features
        self.num_continuous_features = len(self.continuous_features)
        
        self.num_categories = data[self.categorical_features].nunique(axis=0).to_list()

        data = data[self.features] # select features for training
           
        
        # Decide that which features are continuous or categorical by nunique()
        self.one_hot_max_sizes = [1]*self.num_continuous_features + self.num_categories
        
        # Generating missing values
        data = self.missing(data)


        self.norm_mean, self.norm_std = self.compute_normalization(
            data, 
            self.one_hot_max_sizes
        )
        data = self.transform(
            data, 
            self.norm_mean, 
            self.norm_std
        )
        
        self.data = data

        self.EncodedInfo = EncodedInfo(
            len(self.features), self.num_continuous_features, self.num_categories, self.one_hot_max_sizes
        )
        
    def compute_normalization(self, data, one_hot_max_sizes):
        """
        Compute the normalization parameters (i. e. mean to subtract and std
        to divide by) for each feature of the dataset.
        For categorical features mean is zero and std is one.
        i-th feature is denoted to be categorical if one_hot_max_sizes[i] >= 2.
        Returns two vectors: means and stds.
        """
        data = torch.from_numpy(np.array(data)).float()
        
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

    def transform(self, data, norm_mean, norm_std):
        data = torch.from_numpy(np.array(data)).float()

        norm_std = torch.max(norm_std, torch.tensor(1e-9))
        
        data -= norm_mean[None] 
        data /= norm_std[None]

        return data

    def missing(self, X, miss_rate=0.3, mechanism='MAR'):
        X_miss = X.copy()
        n, d = X.shape

        if mechanism == 'MAR':
            prob = 1 / (1 + np.exp(-2 * X['x2']))
            mask_x1 = np.random.rand(n) < (prob * miss_rate)
            mask = np.zeros((n, d), dtype=bool)
            mask[:, 0] = mask_x1  # only x1 has missingness
        else:
            raise NotImplementedError("Only MAR supported")

        self.mask = mask  # 저장

        X_np = X_miss.values
        X_np[mask] = np.nan
        X_miss.iloc[:, :] = X_np

        return X_miss
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])
#%%
def evaluate_mean_and_se(x, true_mean):
    est = np.mean(x)
    std = np.std(x, ddof=1) / np.sqrt(len(x))
    ci_lower = est - 1.96 * std
    ci_upper = est + 1.96 * std
    coverage = ci_lower <= true_mean <= ci_upper
    return est, std, (ci_lower, ci_upper), coverage
#%%
def main():
    #%%
    config = vars(get_args(debug=True)) # default configuration
    set_random_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)
    wandb.config.update(config)
    #%%
    dataset = CustomDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'])
    print(dataset.EncodedInfo)
    #%%
    """model"""
    model_module = importlib.import_module('modules.model')
    importlib.reload(model_module)
    networks = model_module.get_imputation_networks(
        dataset.EncodedInfo.one_hot_max_sizes
    )

    model = model_module.VAEAC(
        config,
        networks,
        device
    ).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    #%%
    count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = count_parameters(model)
    print(f"Number of Parameters: {num_params / 1000:.1f}K")
    #%%
    """train"""
    train_module = importlib.import_module('modules.train')
    importlib.reload(train_module)
    train_module.train_function(
        model,
        networks,
        config,
        optimizer,
        dataloader,
        # valid_dataloader,
        device,
        verbose=True
    )
    #%%
    """listwise"""
    _, true_mean_x1 = generate_data()
    df = pd.DataFrame(dataset.data, columns=['x1', 'x2', 'x3'])
    observed_x1 = df['x1'].dropna().values
    obs_est, obs_se, obs_ci, obs_cov = evaluate_mean_and_se(observed_x1, true_mean_x1)
    
    #%%
    """pairwise"""
    # Pairwise deletion (공분산 기반 회귀)
    df_pairwise = df.copy()
    cov = np.cov(df_pairwise.dropna(subset=['x1'])[['x1', 'x2']], rowvar=False)
    mean_x1 = np.nanmean(df_pairwise['x1'])
    pairwise_est = mean_x1
    pairwise_se = np.nanstd(df_pairwise['x1'], ddof=1) / np.sqrt(df_pairwise['x1'].notna().sum())
    pairwise_ci = (pairwise_est - 1.96 * pairwise_se, pairwise_est + 1.96 * pairwise_se)
    pairwise_cov = pairwise_ci[0] <= true_mean_x1 <= pairwise_ci[1]
    #%%
    """mean"""
    from sklearn.impute import SimpleImputer
    mean_imputer = SimpleImputer(strategy='mean')
    x1_mean_imputed = mean_imputer.fit_transform(df[['x1']])[:, 0]
    mean_est, mean_se, mean_ci, mean_cov = evaluate_mean_and_se(x1_mean_imputed, true_mean_x1)
    #%%
    """regression"""
    from sklearn.linear_model import LinearRegression

    reg_data = df.copy()
    notna_mask = ~reg_data['x1'].isna()
    model_lr = LinearRegression()
    model_lr.fit(reg_data.loc[notna_mask, ['x2']], reg_data.loc[notna_mask, 'x1'])
    
    pred = model_lr.predict(reg_data[['x2']])
    reg_x1 = reg_data['x1'].copy()
    reg_x1[reg_x1.isna()] = pred[reg_x1.isna()]
    
    reg_est, reg_se, reg_ci, reg_cov = evaluate_mean_and_se(reg_x1.values, true_mean_x1)
    #%%
    """Stochastic"""
    # Stochastic regression imputation
    resid = reg_data.loc[notna_mask, 'x1'] - model_lr.predict(reg_data.loc[notna_mask, ['x2']])
    resid_std = resid.std(ddof=1)

    stochastic_x1 = reg_data['x1'].copy()
    rand_noise = np.random.normal(0, resid_std, size=len(stochastic_x1))
    stochastic_x1[stochastic_x1.isna()] = pred[stochastic_x1.isna()] + rand_noise[stochastic_x1.isna()]

    stochastic_est, stochastic_se, stochastic_ci, stochastic_cov = evaluate_mean_and_se(
        stochastic_x1.values, true_mean_x1
    )
    #%%
    """LOCF"""
    # LOCF
    import pandas as pd
    locf_x1 = df['x1'].copy()
    locf_x1 = locf_x1.ffill()
    locf_est, locf_se, locf_ci, locf_cov = evaluate_mean_and_se(locf_x1.values, true_mean_x1)
    #%%
    """Indicator"""
    # Indicator method (x1 결측 여부를 indicator로 추가)
    from sklearn.linear_model import LinearRegression

    indicator_df = df.copy()
    indicator_df['x1_imp'] = df['x1'].copy()
    indicator_df['r_x1'] = df['x1'].isna().astype(int)
    mean_x1 = indicator_df['x1_imp'].mean()  # 임시 채움 (평균)
    indicator_df['x1_imp'].fillna(mean_x1, inplace=True)

    model_indicator = LinearRegression()
    model_indicator.fit(indicator_df[['x1_imp', 'r_x1']], df['x2'])  # 예: x2 ~ x1 + r_x1

    indicator_est, indicator_se, indicator_ci, indicator_cov = evaluate_mean_and_se(
        indicator_df['x1_imp'].values, true_mean_x1
    )

    #%%
    """VAEAC with Multiple Imputation"""
    model.eval()
    M = 50
    imputed_x1_samples = []

    full_imputed = model.impute(dataset, M=M, seed=0) 
    for imputed in tqdm(full_imputed, desc="evaluation"):
        imputed = pd.DataFrame(imputed, columns=dataset.features)
        imputed_x1_samples.append(imputed['x1'])

    imputed_x1_samples = np.array(imputed_x1_samples)  # [M, N]
    
    # Rubin's Rules
    mi_point_estimates = np.mean(imputed_x1_samples, axis=1)
    pooled_mean = np.mean(mi_point_estimates)  # \bar{Q}
    within_var = np.mean(np.var(imputed_x1_samples, axis=1, ddof=1))  # Ū
    between_var = np.var(mi_point_estimates, ddof=1)  # B
    total_var = within_var + (1 + 1/M) * between_var  # T
    
    pooled_se = np.sqrt(total_var)
    pooled_ci = (pooled_mean - 1.96 * pooled_se, pooled_mean + 1.96 * pooled_se)
    pooled_cov = pooled_ci[0] <= true_mean_x1 <= pooled_ci[1]
    
    wandb.log({
        "True Mean of x1": true_mean_x1,
        
        "Pairwise/Estimate": pairwise_est,
        "Pairwise/SE": pairwise_se,
        "Pairwise/CI Lower": pairwise_ci[0],
        "Pairwise/CI Upper": pairwise_ci[1],
        "Pairwise/Coverage": pairwise_cov,

        "Regression/Estimate": reg_est,
        "Regression/SE": reg_se,
        "Regression/CI Lower": reg_ci[0],
        "Regression/CI Upper": reg_ci[1],
        "Regression/Coverage": reg_cov,

        "Stochastic/Estimate": stochastic_est,
        "Stochastic/SE": stochastic_se,
        "Stochastic/CI Lower": stochastic_ci[0],
        "Stochastic/CI Upper": stochastic_ci[1],
        "Stochastic/Coverage": stochastic_cov,

        "LOCF/Estimate": locf_est,
        "LOCF/SE": locf_se,
        "LOCF/CI Lower": locf_ci[0],
        "LOCF/CI Upper": locf_ci[1],
        "LOCF/Coverage": locf_cov,

        "Indicator/Estimate": indicator_est,
        "Indicator/SE": indicator_se,
        "Indicator/CI Lower": indicator_ci[0],
        "Indicator/CI Upper": indicator_ci[1],
        "Indicator/Coverage": indicator_cov,
        
        "Listwise/Estimate": obs_est,
        "Listwise/SE": obs_se,
        "Listwise/CI Lower": obs_ci[0],
        "Listwise/CI Upper": obs_ci[1],
        "Listwise/Coverage": obs_cov,

        "MeanImpute/Estimate": mean_est,
        "MeanImpute/SE": mean_se,
        "MeanImpute/CI Lower": mean_ci[0],
        "MeanImpute/CI Upper": mean_ci[1],
        "MeanImpute/Coverage": mean_cov,

        "VAEAC/MI/Estimate": pooled_mean,
        "VAEAC/MI/SE": pooled_se,
        "VAEAC/MI/CI Lower": pooled_ci[0],
        "VAEAC/MI/CI Upper": pooled_ci[1],
        "VAEAC/MI/Coverage": pooled_cov
    })
    
    
    """model save"""
    base_name = f"{config['model']}_{config['missing_type']}_{config['missing_rate']}_{config['dataset']}"
    model_dir = f"./assets/models/{base_name}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{base_name}_{config['seed']}"
    torch.save(model.state_dict(), f"./{model_dir}/{model_name}.pth")
    artifact = wandb.Artifact(
        "_".join(model_name.split("_")[:-1]), 
        type='model',
        metadata=config) 
    artifact.add_file('./motivation.py')
    wandb.log_artifact(artifact)
    #%%
    wandb.config.update(config, allow_val_change=True)
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%