#%%
import pandas as pd
from scipy.io import arff
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


    return data, continuous_features, categorical_features, integer_features, ClfTarget
