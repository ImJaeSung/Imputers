#%%
import numpy as np
import pandas as pd
from scipy.io import loadmat

#%%
def load_raw_data(dataset):
    if dataset == "abalone":
        data = pd.read_csv('./data/abalone.data', header=None)
        columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "Rings",
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("Sex")
        columns.remove("Rings")
        continuous_features = columns
        categorical_features = [
            "Sex",
            "Rings"
        ]
        integer_features = []
        ClfTarget = "Rings"
        
    elif dataset == "anuran":
        data = pd.read_csv('./data/Frogs_MFCCs.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [x for x in data.columns if x.startswith("MFCCs_")]
        categorical_features = [
            'Family',
            'Genus',
            'Species'
        ]
        integer_features = []
        ClfTarget = "Species"
    
    elif dataset == "banknote":
        data = pd.read_csv('./data/data_banknote_authentication.txt', header=None)
        data.columns = ["variance", "skewness", "curtosis", "entropy", "class"]
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            "variance", "skewness", "curtosis", "entropy"
        ]
        categorical_features = [
            'class',
        ]
        integer_features = []
        ClfTarget = "class"
        
    elif dataset == "breast":
        data = pd.read_csv('./data/wdbc.data', header=None)
        data = data.drop(columns=[0]) # drop ID number
        columns = ["Diagnosis"]
        common_cols = [
            "radius",
            "texture",
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave points",
            "symmetry",
            "fractal dimension",
        ]
        columns += [f"{x}1" for x in common_cols]
        columns += [f"{x}2" for x in common_cols]
        columns += [f"{x}3" for x in common_cols]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = []
        continuous_features += [f"{x}1" for x in common_cols]
        continuous_features += [f"{x}2" for x in common_cols]
        continuous_features += [f"{x}3" for x in common_cols]
        categorical_features = [
            "Diagnosis"
        ]
        integer_features = []
        ClfTarget = "Diagnosis"
        
    elif dataset == "concrete":
        data = pd.read_csv('./data/Concrete_Data.csv')
        columns = [
            "Cement",
            "Blast Furnace Slag",
            "Fly Ash",
            "Water",
            "Superplasticizer",
            "Coarse Aggregate",
            "Fine Aggregate",
            "Age",
            "Concrete compressive strength"
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("Age")
        continuous_features = columns
        categorical_features = [
            "Age",
        ]
        integer_features = []
        ClfTarget = "Age"
        
    elif dataset == "kings":
        data = pd.read_csv('./data/kc_house_data.csv')
        
        continuous_features = [
            'price', 
            'sqft_living',
            'sqft_lot',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'lat',
            'long',
            'sqft_living15',
            'sqft_lot15',
        ]
        categorical_features = [
            'bedrooms',
            'bathrooms',
            'floors',
            'waterfront',
            'view',
            'condition',
            'grade', 
        ]
        integer_features = [
            'price',
            'sqft_living',
            'sqft_lot',
            'sqft_above',
            'sqft_basement',
            'yr_built',
            'yr_renovated',
            'sqft_living15',
            'sqft_lot15',
        ]
        ClfTarget = "grade"
        
    elif dataset == "letter":
        data = pd.read_csv('./data/letter-recognition.data', header=None)
        columns = [
            "lettr",
            "x-box",
            "y-box",
            "width",
            "high",
            "onpix",
            "x-bar",
            "y-bar",
            "x2bar",
            "y2bar",
            "xybar",
            "x2ybr",
            "xy2br",
            "x-ege",
            "xegvy",
            "y-ege",
            "yegvx",
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("lettr")
        continuous_features = columns
        categorical_features = [
            "lettr"
        ]
        integer_features = columns
        ClfTarget = "lettr"
        
    elif dataset == "loan":
        data = pd.read_csv('./data/Bank_Personal_Loan_Modelling.csv')
        
        continuous_features = [
            'Age',
            'Experience',
            'Income', 
            'CCAvg',
            'Mortgage',
        ]
        categorical_features = [
            'Family',
            'Personal Loan',
            'Securities Account',
            'CD Account',
            'Online',
            'CreditCard'
        ]
        integer_features = [
            'Age',
            'Experience',
            'Income', 
            'Mortgage'
        ]
        data = data[continuous_features + categorical_features]
        data = data.dropna()
        ClfTarget = "Personal Loan"
        
    elif dataset == "redwine":
        data = pd.read_csv('./data/winequality-red.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = [
            "quality"
        ]
        integer_features = []
        ClfTarget = "quality"
        
    elif dataset == "whitewine":
        data = pd.read_csv('./data/winequality-white.csv', delimiter=";")
        columns = list(data.columns)
        columns.remove("quality")
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = columns
        categorical_features = [
            "quality"
        ]
        integer_features = []
        ClfTarget = "quality"
        
    elif dataset == "toxicity":
        base = loadmat("./data/TOX-171.mat")
        base = np.concatenate([base['X'], base['Y']], axis=1)
        
        data = pd.DataFrame(base)
        columns = [f"col{i}" for i in range(base.shape[1]-1)]
        columns += ["class"]
        data.columns = columns
        
        columns.remove("class")
        continuous_features = columns
        categorical_features = ["class"]
        integer_features = []
        
        ClfTarget = "class"
        
    elif dataset == "cll":
        base = loadmat("./data/CLL_SUB_111.mat")
        base = np.concatenate([base['X'], base['Y']], axis=1)
        
        data = pd.DataFrame(base)
        columns = [f"col{i}" for i in range(base.shape[1]-1)]
        columns += ["class"]
        data.columns = columns
        
        columns.remove("class")
        continuous_features = columns
        categorical_features = ["class"]
        integer_features = []
        
        ClfTarget = "class"
        
    elif dataset == "orl":
        base = loadmat("./data/ORL.mat")
        base = np.concatenate([base['X'], base['Y']], axis=1)
        
        data = pd.DataFrame(base)
        columns = [f"col{i}" for i in range(base.shape[1]-1)]
        columns += ["class"]
        data.columns = columns
        
        columns.remove("class")
        continuous_features = columns
        categorical_features = ["class"]
        integer_features = []

        ClfTarget = "class"
        
    elif dataset == "glioma":
        base = loadmat("./data/GLIOMA.mat")
        base = np.concatenate([base['X'], base['Y']], axis=1)
        
        data = pd.DataFrame(base)
        columns = [f"col{i}" for i in range(base.shape[1]-1)]
        columns += ["class"]
        data.columns = columns
        
        columns.remove("class")
        continuous_features = columns
        categorical_features = ["class"]
        integer_features = []

        ClfTarget = "class"

    elif dataset == "yale":
        base = loadmat("./data/Yale.mat")
        base = np.concatenate([base['X'], base['Y']], axis=1)
        
        data = pd.DataFrame(base)
        columns = [f"col{i}" for i in range(base.shape[1]-1)]
        columns += ["class"]
        data.columns = columns
        
        columns.remove("class")
        continuous_features = columns
        categorical_features = ["class"]
        integer_features = []

        ClfTarget = "class"
        
    elif dataset == "warppie":
        base = loadmat("./data/warpPIE10P.mat")
        base = np.concatenate([base['X'], base['Y']], axis=1)
        
        data = pd.DataFrame(base)
        columns = [f"col{i}" for i in range(base.shape[1]-1)]
        columns += ["class"]
        data.columns = columns
        
        columns.remove("class")
        continuous_features = columns
        categorical_features = ["class"]
        integer_features = []

        ClfTarget = "class"        
        
        
    return data, continuous_features, categorical_features, integer_features, ClfTarget