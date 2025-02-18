from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge,Lasso
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle
import streamlit as st

from sklearn.neural_network import MLPClassifier

class RegressionMLModeling:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def compute_ML(self):
        if data_type == "regression":
            preds = self.execute_regression_pipeline()
        else:
            preds = []
        return preds, data_type
    
    
    def train_model(self, selected_models, models):
        preds = dict()
        print(selected_models)
        for key in selected_models:
            model = models[key]
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            preds[key] = y_pred
        return preds


    def execute_regression_pipeline(self):
        models = ['Linear Regression', 
                  'Ridge', 
                  'Lasso', 
                  'SGDRegressor',
                  'ARDRegression',
                  'Decision Tree Regressor',
                  'SVR',
                  'AdaBoostRegressor',
                  'KNeighborsRegressor'
                  'Random Forest Regressor']
        
        st.sidebar.markdown("#### Select Regression Models:")
        select_all_toggle = st.sidebar.toggle(label="Select All", key="Select all regression models")
        selected_models = []

        if select_all_toggle:
            selected_models = models
            for model in models:
                st.sidebar.checkbox(model, value=True) 
        else:
            for model in models:
                selected = st.sidebar.checkbox(model)
                if selected:
                    selected_models.append(model)
    
        models = {
                'Linear Regression': LinearRegression(),
                'Ridge' : Ridge(),
                'Lasso' : Lasso(),
                'SGDRegressor' : SGDRegressor(),
                'ARDRegression' : ARDRegression(),
                'Decision Tree Regressor' : DecisionTreeRegressor(),
                'SVR' : SVR(),
                'AdaBoostRegressor' : AdaBoostRegressor(),
                'KNeighborsRegressor' : KNeighborsRegressor(),
                'Random Forest Regressor' : RandomForestRegressor() 
                }
        return self.train_model(selected_models, models)

