#%%
#%%
import pandas as pd
import numpy as np
from scipy.io import arff
from scipy.io import loadmat
#%%
def load_raw_data(config):
    if config["dataset"] == "abalone":
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

    elif config["dataset"] == "anuran":
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
    
    elif config["dataset"] == "banknote":
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
        
    elif config["dataset"] == "breast":
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
        
    elif config["dataset"] == "concrete":
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
        
    elif config["dataset"] == "kings":
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
        
    elif config["dataset"] == "letter":
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
        
    elif config["dataset"] == "loan":
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
        
    elif config["dataset"] == "redwine":
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
        
    elif config["dataset"] == "spam":
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
        
    elif config["dataset"] == "whitewine":
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
        
    # elif config["dataset"] == "yeast":
    #     data = pd.read_csv('./data/yeast.data', delimiter="  ", index_col=False, header=None)
    #     columns = [
    #         "Sequence Name",
    #         "mcg",
    #         "gvh",
    #         "alm",
    #         "mit",
    #         "erl",
    #         "pox",
    #         "vac",
    #         "nuc",
    #         "localization_site"
    #     ]
    #     data.columns = columns
    #     data = data.drop(columns=["Sequence Name"])
        
    #     data = data.dropna() # remove 5 rows
    #     assert data.isna().sum().sum() == 0
        
    #     continuous_features = [
    #         "mcg",
    #         "gvh",
    #         "alm",
    #         "mit",
    #         "erl",
    #         "pox",
    #         "vac",
    #         "nuc",
    #     ]
    #     categorical_features = [
    #         "localization_site"
    #     ]
    #     integer_features = []
    #     ClfTarget = "localization_site"

    elif config["dataset"] == "covtype":
        data = pd.read_csv('./data/covtype.csv')
        # data = data.sample(frac=1, random_state=0).reset_index(drop=True)
        # data = data.dropna(axis=0)
        # data = data.iloc[:50000]
        
        continuous_features = [
            'Elevation',
            'Aspect', 
            'Slope',
            'Horizontal_Distance_To_Hydrology', 
            'Vertical_Distance_To_Hydrology',
            'Horizontal_Distance_To_Roadways',
            'Hillshade_9am',
            'Hillshade_Noon',
            'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points',
        ]
        categorical_features = [
            'Cover_Type',
        ]
        integer_features = continuous_features
        ClfTarget = "Cover_Type"
    
    # train_data, test_data = train_test_split(
    #         data, test_size=config["test_size"], random_state=config["seed"])
    # data = train_data if train else test_data
    # data = data.reset_index(drop=True)

    ########################################################
    #--------------------High-Dimensional-----------------=#
    ########################################################
    elif config["dataset"] == "speed":
        data, _ = arff.loadarff('./data/speeddating.arff') # output : data, meta
        data = pd.DataFrame(data)
        
        for column in data.select_dtypes([object]).columns:
            data[column] = data[column].str.decode('utf-8') # object decoding
        
        data = data.dropna(axis=0) # remove
        assert data.isna().sum().sum() == 0

        common_cols = [
            'd_age',
            'importance_same_race',
            'importance_same_religion',
            'pref_o_attractive',
            'pref_o_sincere',
            'pref_o_intelligence', 
            'pref_o_funny',
            'pref_o_ambitious',
            'pref_o_shared_interests',
            'attractive_o',
            'sinsere_o',
            'intelligence_o',
            'funny_o',
            'ambitous_o', 
            'shared_interests_o',
            'attractive_important', 
            'sincere_important',
            'intellicence_important',
            'funny_important',
            'ambtition_important',
            'shared_interests_important',
            'attractive',
            'sincere',  
            'intelligence',  
            'funny',
            'ambition', 
            'attractive_partner', 
            'sincere_partner',
            'intelligence_partner',  
            'funny_partner',
            'ambition_partner', 
            'shared_interests_partner',
            'sports',
            'tvsports',  
            'exercise', 
            'dining',  
            'museums',  
            'art',  
            'hiking',  
            'gaming',  
            'clubbing',  
            'reading',  
            'tv',  
            'theater',  
            'movies',  
            'concerts',  
            'music',  
            'shopping',  
            'yoga',
            'interests_correlate',
            'expected_happy_with_sd_people',
            'expected_num_interested_in_me',
            'expected_num_matches',
            'like',
            'guess_prob_liked',
        ] 

        continuous_features = [
            'wave',
            'age',
            'age_o',
            'met',
        ]  

        categorical_features = [
            'match',
            'has_null',
            'gender',
            'race',  
            'race_o',
            'samerace',
            'field',
            'decision',
            'decision_o',
        ]
        
        continuous_features += [x for x in common_cols]
        categorical_features += [f"d_{x}" for x in common_cols]
        integer_features = continuous_features

        ClfTarget = "match"

    elif config["dataset"] == "nomao":
        data, _ = arff.loadarff('./data/phpDYCOet.arff') # output : data, meta
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
    
    elif config["dataset"] == "musk":
        data = pd.read_csv('./data/clean2.data', header=None)
        assert data.isna().sum().sum() == 0
            
        column = [i for i in range(1, 167)]
        columns = [
            'molecule_name', 
            'conformation_name'
        ] + [
            f"f{x}" for x in column
        ] + [
            'class'
        ]

        data.columns = columns
        columns.remove('class') 
        columns.remove('molecule_name') 
        columns.remove('conformation_name')
        
        continuous_features = columns
        categorical_features = [
            'class', 
            'molecule_name', 
            'conformation_name'
        ]
        integer_features = continuous_features
         
        ClfTarget = 'class'
    
    elif config["dataset"] == "hillvalley":
        w_noise = pd.read_csv('./data/Hill_Valley_with_noise_Training.data')
        wo_noise = pd.read_csv('./data/Hill_Valley_without_noise_Training.data')
        data = pd.concat([w_noise, wo_noise], axis=0)
        data = data.reset_index(drop=True)

        continuous = [i for i in range(1, 101)]
        continuous_features = [f"X{i}" for i in continuous]
        categorical_features = ["class"]
        integer_features = []

        ClfTarget = 'class'
    
    elif config["dataset"] == "yeast":
        data, _ = arff.loadarff('./data/phpHvRukp.arff') # output : data, meta
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
    
    ########################################################
    #--------------------High-Dimensional-----------------=#
    ########################################################
    elif config["dataset"] == "madelon":
        data, _ = arff.loadarff('./data/phpfLuQE4.arff') # output : data, meta
        data = pd.DataFrame(data)
        
        assert data.isna().sum().sum() == 0

        for column in data.select_dtypes([object]).columns:
            data[column] = data[column].str.decode('utf-8') # object decoding
        
        continuous = [i for i in range(1, 501)]
        continuous_features = [f"V{x}" for x in continuous]
        
        categorical_features = ["Class"]
        integer_features = continuous_features
         
        ClfTarget = 'Class'


    elif config["dataset"] == "bioresponse":
        data, _ = arff.loadarff('./data/phpSSK7iA.arff') # output : data, meta
        data = pd.DataFrame(data)
        
        assert data.isna().sum().sum() == 0

        for column in data.select_dtypes([object]).columns:
            data[column] = data[column].str.decode('utf-8') # object decoding
        
        continuous = [i for i in range(1, 1777)]
        continuous_features = [f"D{x}" for x in continuous]
        
        categorical_features = ["target"]
        integer_features = []
         
        ClfTarget = 'target'
        
    elif config["dataset"] == "titanic":
        data = pd.read_csv('./data/spaceship-titanic.csv')
        data = data.dropna(axis=0) # drop 2087 observations
        
        assert data.isna().sum().sum() == 0
        
        continuous_features = [
            "Age",
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck"
        ]
        
        categorical_features = [
            "HomePlanet",
            "CryoSleep",
            "Cabin",
            "Destination",
            "VIP",
            "Name",
            "Transported"
        ]
        
        integer_features = continuous_features
        
        ClfTarget = "Transported"
    
    elif config["dataset"] == "icu":
        data = pd.read_csv('./data/icu.csv')
        
        assert data.isna().sum().sum() == 0
        
        columns = list(data.columns)
        columns.remove("Unit")
        
        continuous_features = columns
        
        categorical_features = [
            "Unit"
        ]
        
        integer_features = continuous_features
        
        ClfTarget = "Unit"
        
    elif config["dataset"] == "acsincome":
        data = pd.read_csv('./data/acsincome.csv')
        
        assert data.isna().sum().sum() == 0

        continuous_features = [
            "Age",
            "Usual hours worked per week past 12 months"
        ]
        
        categorical_features = [
            "Class of worker",
            "Educational attainment",
            "Marital status",
            "Occupation",
            "Place of birth",
            "Sex",
            "Recoded race",
            "Income"
        ]
        
        integer_features = continuous_features
        
        ClfTarget = "Income"

    elif config["dataset"] == "acstravel":
        data = pd.read_csv('./data/acstravel.csv')
        data = data.dropna(axis=0)
        
        assert data.isna().sum().sum() == 0

        columns = list(data.columns)
        columns.remove("Age")
        
        continuous_features = [
            "Age"
        ]
        
        categorical_features = columns        
        integer_features = continuous_features
        
        ClfTarget = "Travel Time to Work"

    #####   
    elif config["dataset"] == "parkinson":
        base = pd.read_csv("./data/pd_speech_features.csv").iloc[:, 1:] # id column
        columns = list(base.iloc[0])
        
        data = base.iloc[1:].reset_index(drop=True) # column name, id delete
        data.columns = columns
        data = data.astype(float)

        columns.remove('gender')
        columns.remove('class')
        
        continuous_features = columns
        categorical_features = [
            'gender',
            'class'
        ]
        integer_features = [
            'numPulses',
            'numPeriodsPulses',
        ]
        ClfTarget = 'class'
        
    elif config["dataset"] == "arrhythmia":
        base = pd.read_csv("./data/arrhythmia.data",skiprows=1, header=None)
        base = base[(base == '?').sum(axis=1) == 0]
        data = base.dropna()
        data = data.astype(float)
        
        # manually define
        columns = [
            "age", 
            "sex", 
            "height", 
            "weight", 
            "qrs_duration", 
            "pr_interval",
            "qt_interval",
            "t_interval", 
            "p_interval",
            "qrs_angle",
            "t_angle",
            "p_angle", 
            "qrst_angle",
            "j_angle",
            "heart_rate"
        ]

        # 16~159: channel DI ~ V6
        channels = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        features = [
            "q_wave_width",
            "r_wave_width", 
            "s_wave_width", 
            "r_prime_wave_width", 
            "s_prime_wave_width",
            "num_intrinsic_deflections",
            "ragged_r_wave", 
            "diphasic_r_wave", 
            "ragged_p_wave", 
            "diphasic_p_wave",
            "ragged_t_wave", 
            "diphasic_t_wave"
        ]
        for ch in channels:
            for f in features:
                columns.append(f"{ch.lower()}_{f}")

        # 160~279: channel DI ~ V6
        amplitude_features = [
            "jj_wave_amp", 
            "q_wave_amp", 
            "r_wave_amp", 
            "s_wave_amp", 
            "r_prime_wave_amp", 
            "s_prime_wave_amp",
            "p_wave_amp", 
            "t_wave_amp", 
            "qrsa", 
            "qrsta"
        ]
        for ch in channels:
            for f in amplitude_features:
                columns.append(f"{ch.lower()}_{f}")
        
        columns += ["Class"]
        
        assert data.shape[1] == len(columns)
        
        data.columns = columns
        categorical_features = [
            'sex',
            'di_ragged_r_wave',
            'di_diphasic_r_wave',
            'di_ragged_p_wave',
            'di_diphasic_p_wave',
            'di_ragged_t_wave',
            'di_diphasic_t_wave',
            'Class'
        ]
        
        continuous_features = [
            f for f in columns if f not in categorical_features
        ]
        
        integer_features [
            'age',
            'height',
            'weight',
            'qrs_duration',
            'pr_interval',
            'qt_interval',
            't_interval',
            'p_interval',
            'qrs_angle',
            't_angle',
            'p_angle',
            'qrst_angle',
            'j_angle',
            'heart_rate',
            'di_q_wave_width',
            'di_r_wave_width',
            'di_s_wave_width',
            'di_r_prime_wave_width',
            'di_s_prime_wave_width',
            'di_num_intrinsic_deflections',
            'di_ragged_r_wave',
            'di_diphasic_r_wave',
            'di_ragged_p_wave',
            'di_diphasic_p_wave',
            'di_ragged_t_wave',
            'di_diphasic_t_wave',
            'dii_q_wave_width',
            'dii_r_wave_width',
            'dii_s_wave_width',
            'dii_r_prime_wave_width',
            'dii_s_prime_wave_width',
            'dii_num_intrinsic_deflections',
            'dii_ragged_r_wave',
            'dii_diphasic_r_wave',
            'dii_ragged_p_wave',
            'dii_diphasic_p_wave',
            'dii_ragged_t_wave',
            'dii_diphasic_t_wave',
            'diii_q_wave_width',
            'diii_r_wave_width',
            'diii_s_wave_width',
            'diii_r_prime_wave_width',
            'diii_s_prime_wave_width',
            'diii_num_intrinsic_deflections',
            'diii_ragged_r_wave',
            'diii_diphasic_r_wave',
            'diii_ragged_p_wave',
            'diii_diphasic_p_wave',
            'diii_ragged_t_wave',
            'diii_diphasic_t_wave',
            'avr_q_wave_width',
            'avr_r_wave_width',
            'avr_s_wave_width',
            'avr_r_prime_wave_width',
            'avr_s_prime_wave_width',
            'avr_num_intrinsic_deflections',
            'avr_ragged_r_wave',
            'avr_diphasic_r_wave',
            'avr_ragged_p_wave',
            'avr_diphasic_p_wave',
            'avr_ragged_t_wave',
            'avr_diphasic_t_wave',
            'avl_q_wave_width',
            'avl_r_wave_width',
            'avl_s_wave_width',
            'avl_r_prime_wave_width',
            'avl_s_prime_wave_width',
            'avl_num_intrinsic_deflections',
            'avl_ragged_r_wave',
            'avl_diphasic_r_wave',
            'avl_ragged_p_wave',
            'avl_diphasic_p_wave',
            'avl_ragged_t_wave',
            'avl_diphasic_t_wave',
            'avf_q_wave_width',
            'avf_r_wave_width',
            'avf_s_wave_width',
            'avf_r_prime_wave_width',
            'avf_s_prime_wave_width',
            'avf_num_intrinsic_deflections',
            'avf_ragged_r_wave',
            'avf_diphasic_r_wave',
            'avf_ragged_p_wave',
            'avf_diphasic_p_wave',
            'avf_ragged_t_wave',
            'avf_diphasic_t_wave',
            'v1_q_wave_width',
            'v1_r_wave_width',
            'v1_s_wave_width',
            'v1_r_prime_wave_width',
            'v1_s_prime_wave_width',
            'v1_num_intrinsic_deflections',
            'v1_ragged_r_wave',
            'v1_diphasic_r_wave',
            'v1_ragged_p_wave',
            'v1_diphasic_p_wave',
            'v1_ragged_t_wave',
            'v1_diphasic_t_wave',
            'v2_q_wave_width',
            'v2_r_wave_width',
            'v2_s_wave_width',
            'v2_r_prime_wave_width',
            'v2_s_prime_wave_width',
            'v2_num_intrinsic_deflections',
            'v2_ragged_r_wave',
            'v2_diphasic_r_wave',
            'v2_ragged_p_wave',
            'v2_diphasic_p_wave',
            'v2_ragged_t_wave',
            'v2_diphasic_t_wave',
            'v3_q_wave_width',
            'v3_r_wave_width',
            'v3_s_wave_width',
            'v3_r_prime_wave_width',
            'v3_s_prime_wave_width',
            'v3_num_intrinsic_deflections',
            'v3_ragged_r_wave',
            'v3_diphasic_r_wave',
            'v3_ragged_p_wave',
            'v3_diphasic_p_wave',
            'v3_ragged_t_wave',
            'v3_diphasic_t_wave',
            'v4_q_wave_width',
            'v4_r_wave_width',
            'v4_s_wave_width',
            'v4_r_prime_wave_width',
            'v4_s_prime_wave_width',
            'v4_num_intrinsic_deflections',
            'v4_ragged_r_wave',
            'v4_diphasic_r_wave',
            'v4_ragged_p_wave',
            'v4_diphasic_p_wave',
            'v4_ragged_t_wave',
            'v4_diphasic_t_wave',
            'v5_q_wave_width',
            'v5_r_wave_width',
            'v5_s_wave_width',
            'v5_r_prime_wave_width',
            'v5_s_prime_wave_width',
            'v5_num_intrinsic_deflections',
            'v5_ragged_r_wave',
            'v5_diphasic_r_wave',
            'v5_ragged_p_wave',
            'v5_diphasic_p_wave',
            'v5_ragged_t_wave',
            'v5_diphasic_t_wave',
            'v6_q_wave_width',
            'v6_r_wave_width',
            'v6_s_wave_width',
            'v6_r_prime_wave_width',
            'v6_s_prime_wave_width',
            'v6_num_intrinsic_deflections',
            'v6_ragged_r_wave',
            'v6_diphasic_r_wave',
            'v6_ragged_p_wave',
            'v6_diphasic_p_wave',
            'v6_ragged_t_wave',
            'v6_diphasic_t_wave'
        ]
        
        ClfTarget = 'Class'
        
        
    # high-dimensional
    elif config["dataset"] == "arcene":
        
        base1 = pd.read_csv('./data/arcene_train.data', header=None, delimiter=" ")
        base2 = pd.read_csv('./data/arcene_valid.data', header=None, delimiter=" ")
        
        columns = [f"col{x}" for x in range(len(base1.columns) - 1)]
        columns += ["class"]
        
        base1.columns = columns 
        base2.columns = columns
         
        base1['class'] = pd.read_csv('./data/arcene_train.labels', header=None, delimiter=" ")
        base2['class'] = pd.read_csv('./data/arcene_valid.labels', header=None, delimiter=" ")
        
        data = pd.concat([base1, base2], axis=0)
        
        columns.remove('class')
        
        continuous_features = columns
        categorical_features = ["class"]
        integer_features = continuous_features
        ClfTarget = "class"
    
    elif config["dataset"] == "lung":
        base = loadmat("./data/lung.mat")
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
    
    elif config["dataset"] == "toxicity":
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
    
    elif config["dataset"] == "prostate":
        base = loadmat("./data/Prostate_GE.mat")
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
        
    elif config["dataset"] == "cll":
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
        
    elif config["dataset"] == "orl":
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
    
    elif config["dataset"] == "allaml":
        base = loadmat("./data/ALLAML.mat")
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
        
    elif config["dataset"] == "glioma":
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

    elif config["dataset"] == "yale":
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
        
    elif config["dataset"] == "warppie":
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

    elif config["dataset"] == "shoppers":
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

    elif config["dataset"] == "default":
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

    elif config["dataset"] == "BAF":
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

# %%
