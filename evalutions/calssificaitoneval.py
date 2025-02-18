
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd


class ClassificationEvaluation:
    def __init__(self, y_test, y_preds, n_train_cols=0, category=None):
        self.y_test = y_test
        self.y_preds = y_preds
        self.category = category
        self.n_train_cols = n_train_cols


    def evaluate(self):
        if self.category == "classification":
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
            results = pd.DataFrame(index=metrics, columns=self.y_preds.keys())

            for model_name, y_pred in self.y_preds.items():
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='macro')
                recall = recall_score(self.y_test, y_pred, average='macro')
                f1 = f1_score(self.y_test, y_pred, average='macro')
                auc = roc_auc_score(self.y_test, y_pred)

                results.loc['Accuracy', model_name] = accuracy
                results.loc['Precision', model_name] = precision
                results.loc['Recall', model_name] = recall
                results.loc['F1', model_name] = f1
                results.loc['ROC_AUC', model_name] = auc

            results.index = results.index.set_names(['Models Name'])
            return results





