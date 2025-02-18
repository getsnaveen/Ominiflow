
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

import streamlit as st


class ClassificationMLModeling:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def compute_ML(self):
        # data_type = st.sidebar.selectbox('ML Model Type', ['none', 'regression', 'classification', 'clustering'])
        if data_type == "classification":
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
        models = ['LogisticRegression', 
                  'SGDClassifier',
                  'DecisionTreeClassifier',
                  'SVC',
                  'GaussianNB',
                  'AdaBoostClassifier',
                  'KNeighborsClassifier'
                  'RandomForestClassifier']
        
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
                'LogisticRegression': LogisticRegression(),
                'SGDClassifier' : SGDClassifier(),
                'DecisionTreeClassifier' : DecisionTreeClassifier(),
                'SVC' : SVC(),
                'GaussianNB' : GaussianNB(),
                'AdaBoostClassifier' : AdaBoostClassifier(),
                'KNeighborsClassifier' : KNeighborsClassifier(),
                'RandomForestClassifier' : RandomForestClassifier() 
                }
        return self.train_model(selected_models, models)

