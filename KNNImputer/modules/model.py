#%%
import numpy as np
from sklearn.impute import KNNImputer
#%%
class KNNI(KNNImputer):
    def __init__(self, 
                 n_neighbors=5, 
                 **kwargs
        ):
        
        super().__init__(
            n_neighbors=n_neighbors,
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
