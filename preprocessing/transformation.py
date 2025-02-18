
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import add_dummy_feature
from sklearn.preprocessing import OrdinalEncoder

# from sklearn.preprocessing import BinaryEncoding
# from sklearn.preprocessing import FrequencyEncoding
import category_encoders as ce
import pandas as pd
import numpy as np
import streamlit as st
import random

class CustomTransfromation:
    def __init__(self):
        pass  

    def execute_data_transformation(self, X, status):
        c1, c2 = st.sidebar.columns(2)
        imputation_strategy_num = c1.selectbox('Numerical', ['none', 
                                                             'standardscaling', 
                                                             'minmaxscaling',
                                                             'normalizing',
                                                             'robustscaling',
                                                             'quantilescaling',
                                                             'boxcoxscaling',
                                                             'logscaling',
                                                             'ploynomialscaling'
                                                             ])

        imputation_strategy_cat = c2.selectbox('Categorical', [ 'none', 
                                                                'labelencoding',
                                                                'onehotencoding',
                                                                'ordinalencoding'
                                                                ])
        num_data = X.select_dtypes(include='number')
        print(f"Num cols shape: {len(num_data.columns)}")

        cat_cols = [col for col in X.columns if col not in num_data.columns]
        print(f"Cat cols shape: {len(cat_cols)}")
        cat_data = X.loc[:, cat_cols]

        if imputation_strategy_num != "none":
            status.markdown("#### Status: Numerical Data Transformation completed.")
            if imputation_strategy_num == 'standardscaling':
                scaling = StandardScaler()
                num_data_imputed = pd.DataFrame(scaling.fit_transform(num_data), columns=num_data.columns)
            elif imputation_strategy_num == 'minmaxscaling':
                scaling = MinMaxScaler()
                num_data_imputed = pd.DataFrame(scaling.fit_transform(num_data), columns=num_data.columns)                
            elif imputation_strategy_num == 'normalizing':
                scaling = Normalizer()
                num_data_imputed = pd.DataFrame(scaling.fit_transform(num_data), columns=num_data.columns)
            elif imputation_strategy_num == 'robustscaling':
                scaling = RobustScaler()
                num_data_imputed = pd.DataFrame(scaling.fit_transform(num_data), columns=num_data.columns)
            elif imputation_strategy_num == 'quantilescaling':
                scaling = QuantileTransformer(n_quantiles=random.randint(1,1000),
                                              output_distribution='normal',
                                                random_state=0)
                num_data_imputed = pd.DataFrame(scaling.fit_transform(num_data), columns=num_data.columns)
            elif imputation_strategy_num == 'boxcoxscaling':
                scaling = PowerTransformer(method='box-cox')
                num_data_imputed = pd.DataFrame(scaling.fit_transform(num_data), columns=num_data.columns)
            elif imputation_strategy_num == 'logscaling':
                scaling = FunctionTransformer(np.log2, validate = True)
                num_data_imputed = pd.DataFrame(scaling.fit_transform(num_data), columns=num_data.columns)
            elif imputation_strategy_num == 'ploynomialscaling':
                scaling = PolynomialFeatures()
                num_data_imputed = pd.DataFrame(scaling.fit_transform(num_data), columns=num_data.columns)
            else :
                status.markdown("#### Status: Invalid Numerical Data Transformation selected.")
            
            
        if imputation_strategy_cat != "none":
            status.markdown("#### Status: Categorical Data Transformation completed.")
            if imputation_strategy_cat == 'labelencoding':
                encoder = LabelEncoder()
                cat_data_imputed = encoder.fit_transform(cat_data.columns)
            elif imputation_strategy_cat == 'onehotencoding':                    
                cat_data_imputed = pd.get_dummies(cat_data, columns=cat_data.columns)
            elif imputation_strategy_cat == 'ordinalencoding':
                encoder = OrdinalEncoder()
                cat_data_imputed = encoder.fit_transform(cat_data.columns)
            else :
                status.markdown("#### Status: Invalid Categorical Data Transformation selected.")
        return pd.concat([num_data_imputed, cat_data_imputed], axis=1)

