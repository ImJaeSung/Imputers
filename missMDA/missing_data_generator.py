#%%
import os
import argparse
import importlib
#%%
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', type=str, default='whitewine', 
                        help="""
                        Dataset options: 
                        abalone, banknote, breast, redwine, whitewine
                        """)
    parser.add_argument("--missing_type", default="MCAR", type=str,
                        help="how to generate missing: MCAR, MAR, MNARL, MNARQ") 
    parser.add_argument('--missing_rate', default=0.3, type=float)
    
    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()

#%%
def main():
#%%    
    config = vars(get_args(debug=False)) # default configuration
    
    assert config["missing_type"] != None
    #%%
    """missing data generation"""
    dataset_module = importlib.import_module('datasets.imputation')
    importlib.reload(dataset_module)
    dataset = dataset_module.CustomDataset(config).df_data
    rawdata = dataset_module.CustomDataset(config).raw_data
    # %%
    """save missing data"""
    data_dir = f"./missing_data/{config['dataset']}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dataset_name = f"{config['dataset']}_{config['missing_type']}_{config['missing_rate']}_{config['seed']}"
    dataset.to_csv(f"./{data_dir}/{dataset_name}.csv", index=False)
    """save true data"""
    data_dir = f"./true_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dataset_name = f"{config['dataset']}"
    rawdata.to_csv(f"./{data_dir}/{dataset_name}.csv", index=False)
#%%
if __name__ == "__main__":
    main()