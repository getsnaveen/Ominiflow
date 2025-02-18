from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_squared_log_error
import numpy as np
import pandas as pd


class RegressionEvaluation:
    def __init__(self, y_test, y_preds, n_train_cols=0, category=None):
        self.y_test = y_test
        self.y_preds = y_preds
        self.category = category
        self.n_train_cols = n_train_cols


    def evaluate(self):
        if self.category == "regression":
            metrics = ['MAE', 'MSE', 'RMSE', 'RMSLE', 'R2', 'adjusted_R2']
            results = pd.DataFrame(index=metrics, columns=self.y_preds.keys())

            for model_name, y_pred in self.y_preds.items():
                mae = mean_absolute_error(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                rmsle = np.sqrt(mean_squared_log_error(self.y_test, y_pred))
                r2 = r2_score(self.y_test, y_pred)
                n = len(self.y_test)
                k = self.n_train_cols
                adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

                results.loc['MAE', model_name] = mae
                results.loc['MSE', model_name] = mse
                results.loc['RMSE', model_name] = rmse
                results.loc['RMSLE', model_name] = rmsle
                results.loc['R2', model_name] = r2
                results.loc['adjusted_R2', model_name] = adjusted_r2
            results = results.transpose()
            results.index = results.index.set_names(['Models Name    '])
            return results


       