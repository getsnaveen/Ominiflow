from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score
import numpy as np
import pandas as pd


class ClusteringEvaluation:
    def __init__(self, y_test, y_preds, n_train_cols=0, category=None):
        self.y_test = y_test
        self.y_preds = y_preds
        self.category = category
        self.n_train_cols = n_train_cols


    def evaluate(self):
        if self.category == "clustering":
            metrics = ['silhouette_score', 'homogeneity_score', 'completeness_score']
            results = pd.DataFrame(index=metrics, columns=self.y_preds.keys())

            for model_name, y_pred in self.y_preds.items():
                silhouette_score = silhouette_score(self.y_test)
                homogeneity_score = homogeneity_score(self.y_test)
                completeness_score = completeness_score(self.y_test)
                results.loc['silhouette_score', model_name] = silhouette_score
                results.loc['homogeneity_score', model_name] = homogeneity_score
                results.loc['completeness_score', model_name] = completeness_score
            results = results.transpose()
            results.index = results.index.set_names(['Models Name    '])
            return results

