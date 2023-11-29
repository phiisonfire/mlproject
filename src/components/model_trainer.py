"""
Train different models
Choose the model that has the best performance
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV
import warnings

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

class ModelTrainer():
    def __init__(self) -> None:
        self.model_trainer_file_path = os.path.join('artifacts', 'model.pkl')
    
    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info('Split train/test array into feature-target arrays')
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1])
            
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor()
            }
            
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test, models=models)
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("None of those model fit well on the data.")
            logging.info('Found best model on testing dataset.')
            
            save_object(
                object=models[best_model_name],
                file_path=self.model_trainer_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)


    