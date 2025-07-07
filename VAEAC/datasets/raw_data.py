#%%
import pandas as pd
import numpy as np
from scipy.io import arff
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
        
    elif dataset == "nomao":
        data, _ = arff.loadarff('./data/nomao.arff') # output : data, meta
        data = pd.DataFrame(data)
        
        assert data.isna().sum().sum() == 0

        for column in data.select_dtypes([object]).columns:
            data[column] = data[column].str.decode('utf-8') # object decoding
        
        categorical = [
            7, 8, 15, 16, 23, 24, 31, 32, 39, 40, 
            47, 48, 55, 56, 63, 64, 71, 72, 79, 80, 
            87,  88, 92, 96, 100, 104, 108, 112, 116 
        ]
        continuous = [i for i in range(1, 119)]
        continuous = [x for x in continuous if x not in categorical]

        continuous_features = [f"V{x}" for x in continuous]
        categorical_features = [f"V{x}" for x in categorical] + ['Class']
        integer_features = []
         
        ClfTarget = 'Class'
        
    elif dataset == "yeast":
        data, _ = arff.loadarff('./data/yeast.arff') # output : data, meta
        data = pd.DataFrame(data)
        
        assert data.isna().sum().sum() == 0

        for column in data.select_dtypes([object]).columns:
            data[column] = data[column].str.decode('utf-8') # object decoding
        
        continuous = [i for i in range(1, 104)]
        continuous_features = [f"attr{x}" for x in continuous]
        
        categorical = [i for i in range(1, 15)]
        categorical_features = [f"class{x}" for x in categorical]
        integer_features = []
         
        ClfTarget = 'class14'
        
    elif dataset == "diabetes":
        data = pd.read_csv('./data/diabetes.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
        categorical_features = ["Outcome"]
        integer_features = continuous_features
        
        ClfTarget = 'Outcome'
        
    elif dataset == "dna":
        data, _ = arff.loadarff('./data/dna.arff') # output : data, meta
        data = pd.DataFrame(data)
        
        assert data.isna().sum().sum() == 0

        for column in data.select_dtypes([object]).columns:
            data[column] = data[column].str.decode('utf-8') # object decoding
        
        continuous_features = []
        
        categorical = [i for i in range(1, 180)]
        categorical_features = [f"A{x}" for x in categorical]
        categorical_features += ["class"]
        
        integer_features = []
         
        ClfTarget = 'class'
    
    elif dataset == "ncbirths":
        data = pd.read_csv('./data/NCbirths.csv')
        data = data.drop(columns=["ID"]) # drop ID number
        
        data.dropna(axis=0, inplace=True)
        
        assert data.isna().sum().sum() == 0
        
        columns = list(data.columns)
        continuous_features = [
            "MomAge",
            "Weeks",
            "Gained",
            "BirthWeightOz",
            "BirthWeightGm",
        ]
        
        for i in continuous_features: 
           columns.remove(i)
        categorical_features = columns
        integer_features = [
            "MomAge", 
            "Weeks", 
            "Gained", 
            "BirthWeightOz"
        ]
        
        ClfTarget = 'MomRace'   
        
    elif dataset == "spam":
        data = pd.read_csv('./data/spambase.data', header=None)
        columns = [
            "word_freq_make",
            "word_freq_address",
            "word_freq_all",
            "word_freq_3d",
            "word_freq_our",
            "word_freq_over",
            "word_freq_remove",
            "word_freq_internet",
            "word_freq_order",
            "word_freq_mail",
            "word_freq_receive",
            "word_freq_will",
            'word_freq_people',
            "word_freq_report",
            'word_freq_addresses',
            "word_freq_free",
            "word_freq_business",
            "word_freq_email",
            "word_freq_you",
            'word_freq_credit',
            'word_freq_your',
            "word_freq_font",
            'word_freq_000',
            'word_freq_money',
            "word_freq_hp",
            'word_freq_hpl',
            'word_freq_george',
            "word_freq_650",
            "word_freq_lab",
            "word_freq_labs",
            'word_freq_telnet',
            "word_freq_857",
            "word_freq_data",
            'word_freq_415',
            "word_freq_85",
            "word_freq_technology",
            "word_freq_1999",
            "word_freq_parts",
            'word_freq_pm',
            "word_freq_direct",
            "word_freq_cs",
            "word_freq_meeting",
            'word_freq_original',
            'word_freq_project',
            'word_freq_re',
            'word_freq_edu',
            'word_freq_table',
            'word_freq_conference',
            "char_freq_;",
            "char_freq_(",
            "char_freq_[",
            "char_freq_!",
            "char_freq_$",
            "char_freq_#",
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
            "class"
        ]
        data.columns = columns
        
        assert data.isna().sum().sum() == 0
        
        columns.remove("class")
        continuous_features = columns
        categorical_features = [
            "class"
        ]
        integer_features = [
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
        ]
        ClfTarget = "class" 
    
    elif dataset == "simulated3":
        data = pd.read_csv('./data/10.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous = [i for i in range(1, 11)]
        continuous_features = [f"con_V{x}" for x in continuous]
        
        categorical = [i for i in range(11, 16)]
        categorical_features = [f"cat_V{x}" for x in categorical]
        
        integer_features = []
         
        ClfTarget = 'cat_V15'
  
    elif dataset == 'simulated2':
        data = pd.read_csv('./data/simulated2.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous = [i for i in range(1, 11)]
        continuous_features = [f"con_V{x}" for x in continuous]
        
        categorical = [i for i in range(11, 16)]
        categorical_features = [f"cat_V{x}" for x in categorical]
        
        integer_features = []
         
        ClfTarget = 'cat_V15'
            
    elif dataset == 'simulated1':
        data = pd.read_csv('./data/simulated2.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous = [i for i in range(1, 11)]
        continuous_features = [f"con_V{x}" for x in continuous]
        
        categorical = [i for i in range(11, 16)]
        categorical_features = [f"cat_V{x}" for x in categorical]
        
        integer_features = []
         
        ClfTarget = 'cat_V15'
        
    elif dataset == "vietnami":
        data = pd.read_csv('./data/vietnami.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous = [
            "pharvis",	
            "lnhhexp",
            "age", 
            "illness", 
            "illdays",	
            "actdays",	
            "commune",
        ]
        continuous_features = [f"con_{x}" for x in continuous]
        
        categorical = [
            "sex",
            "married",	
            "educ",		
            "injury",	
            "insurance",	
        ]
        categorical_features = [f"cat_{x}" for x in categorical]
        
        integer = [
            "pharvis",
            "illness", 
            "illdays",	
            "actdays",	
            "commune",
        ]
        integer_features = [f"con_{x}" for x in integer]
         
        ClfTarget = 'cat_insurance'
        
    elif dataset == "carcinoma":
        data = pd.read_csv('./data/carcinoma.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = []
        
        categorical = ["A", "B", "C", "D", "E", "F", "G"]
        categorical_features = [f"cat_{x}" for x in categorical]
    
        integer_features = []
         
        ClfTarget = 'cat_G'
        
    elif dataset == "dti": 
        data = pd.read_csv('./data/dti.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous = [i for i in range(1, 94)]
        continuous_features = [f"con_cca_{x}" for x in continuous]
        
        categorical_features = []
    
        integer_features = []
         
        ClfTarget = None       
    
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

    elif dataset == "shoppers":
        ### https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
        data = pd.read_csv('./data/online_shoppers_intention.csv')
        
        assert data.isna().sum().sum() == 0

        continuous_features = [
            'Administrative_Duration',   
            'Informational_Duration',      
            'ProductRelated_Duration',     
            'BounceRates',             
            'ExitRates',                   
            'PageValues',                
            'SpecialDay',                
            'Administrative',    
            'Informational',     
            'ProductRelated',      
        ]

        categorical_features = [
            'Month',               
            'VisitorType',         
            'Weekend',          
            'OperatingSystems',    
            'Browser',            
            'Region',           
            'TrafficType',        
            "Revenue"
        ]

        integer_features = [
            'Administrative',    
            'Informational',      
            'ProductRelated',     
        ]

        ClfTarget = "Revenue"

    elif dataset == "default":
        data = pd.read_csv('./data/default.csv')
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            'LIMIT_BAL',  
            'AGE', 
            'BILL_AMT1', 
            'BILL_AMT2',
            'BILL_AMT3',
            'BILL_AMT4', 
            'BILL_AMT5', 
            'BILL_AMT6', 
            'PAY_AMT1',
            'PAY_AMT2', 
            'PAY_AMT3', 
            'PAY_AMT4', 
            'PAY_AMT5', 
            'PAY_AMT6',
        ]
        categorical_features = [
            'SEX', 
            'EDUCATION', 
            'MARRIAGE', 
            'PAY_0',
            'PAY_2', 
            'PAY_3', 
            'PAY_4',
            'PAY_5', 
            'PAY_6', 
            'default_payment_next_month'
        ]
        integer_features = [
            'LIMIT_BAL',  
            'AGE', 
        ]
        ClfTarget = "default_payment_next_month"    

    elif dataset == "BAF":
        # https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022/data
        data = pd.read_csv('./data/BAF.csv')
        
        ### remove missing values
        data = data.loc[data["prev_address_months_count"] != -1]
        data = data.loc[data["current_address_months_count"] != -1]
        data = data.loc[data["intended_balcon_amount"] >= 0]
        data = data.loc[data["bank_months_count"] != -1]
        data = data.loc[data["session_length_in_minutes"] != -1]
        data = data.loc[data["device_distinct_emails_8w"] != -1]
        data = data.reset_index(drop=True)
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            'income', 
            'name_email_similarity',
            'prev_address_months_count', 
            'current_address_months_count',
            'days_since_request', 
            'intended_balcon_amount',
            'zip_count_4w', 
            'velocity_6h', 
            'velocity_24h',
            'velocity_4w', 
            'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w', 
            'credit_risk_score', 
            'bank_months_count',
            'proposed_credit_limit', 
            'session_length_in_minutes', 
        ]
        categorical_features = [
            'customer_age', 
            'payment_type', 
            'employment_status',
            'email_is_free', 
            'housing_status',
            'phone_home_valid', 
            'phone_mobile_valid', 
            'has_other_cards', 
            'foreign_request', 
            'source',
            'device_os', 
            'keep_alive_session',
            'device_distinct_emails_8w', 
            'month',
            'fraud_bool', 
        ]
        integer_features = [
            'prev_address_months_count', 
            'current_address_months_count',
            'zip_count_4w', 
            'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w', 
            'credit_risk_score', 
            'bank_months_count',
        ]
        ClfTarget = "fraud_bool"    
        
    return data, continuous_features, categorical_features, integer_features, ClfTarget