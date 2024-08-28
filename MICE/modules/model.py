#%%
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
#%%
class MICE(IterativeImputer):
    def __init__(self, 
                 max_iter,
                 random_state,
                 **kwargs
        ):
        
        super().__init__(
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X, y=None):
        
        return super().fit(X, y)

    def transform(self, X):
        imputed_data = super().transform(X)
        
        return imputed_data

    def fit_transform(self, X, y=None):
        imputed_data = super().fit_transform(X, y)
        
        return imputed_data
# %%