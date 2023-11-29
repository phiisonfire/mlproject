"""
Module: model_trainer

This module provides a class, ModelTrainer, for training and saving the best regression model based on the given training and testing datasets. 
It includes various regression models such as Linear Regression, Lasso Regression, Ridge Regression, KNeighbors Regressor, RandomForest Regressor, 
and AdaBoost Regressor. The module also contains utility functions for evaluating and saving the best model.

Classes:
    - ModelTrainer: A class for training regression models and saving the best-performing model.

Functions:
    - initiate_model_trainer(train_arr, test_arr, preprocessor_path): 
    Initializes the ModelTrainer class and trains regression models using the provided training and testing arrays.

Usage Example:
    ```
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path)
    ```

Note:
    - The trained model is saved in the 'artifacts' directory with the filename 'model.pkl'.
    - The module requires the 'src.exception', 'src.logger', and 'src.utils' modules for handling exceptions, logging, and utility functions, respectively.
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
            
            params = {
                
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


    