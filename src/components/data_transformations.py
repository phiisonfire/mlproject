"""
Module: data_transformation

This module provides functionality for data transformation and preprocessing.

Classes:
--------
DataTransformation:
    A class for creating a data transformer object for preprocessing numerical and categorical features.

DataTransformationConfig:
    A dataclass that holds configuration parameters for data transformation.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # pipeline for column transformations (multiple steps)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Create and return a data transformer object for preprocessing numerical and categorical features.

    Returns:
    --------
    sklearn.compose.ColumnTransformer:
        A ColumnTransformer object containing two pipelines:
        - 'num_pipeline' for numerical feature preprocessing, including median imputation and standard scaling.
        - 'cat_pipeline' for categorical feature preprocessing, including most frequent imputation, one-hot encoding, and standard scaling.

    Raises:
    -------
    CustomException:
        If an exception occurs during the creation of the data transformer object, it is caught and re-raised with additional information.
    """
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        num_features = ['reading score', 'writing score']
        cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
        
        try:
            num_pipeline = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Created `preprocessing` Pipeline object for numerical features.')
            
            
            cat_pipeline = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Created `preprocessing` Pipeline object for categorical features.')
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )
            
            logging.info('Preprocessing object has been created.')
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads training and testing data from CSV files, performs data transformation,
        and returns transformed training and testing arrays along with the saved preprocessing object.

        Parameters:
        - train_path (str): Path to the CSV file containing the training data.
        - test_path (str): Path to the CSV file containing the testing data.

        Returns:
        Tuple[np.ndarray, np.ndarray, str]: A tuple containing:
            - np.ndarray: Transformed training data array with features and target variable.
            - np.ndarray: Transformed testing data array with features and target variable.
            - str: File path of the saved preprocessing object.

        Raises:
        - CustomException: If an error occurs during the data transformation process.

        The function reads CSV files from the specified paths, initializes a data preprocessor,
        extracts target, numerical, and categorical column names, and splits the data into
        feature and target variables. It then fits and transforms the training data and transforms
        the testing data using the preprocessor. The transformed data arrays are merged with the
        target variable. The preprocessing object is saved, and the function returns the transformed
        data arrays along with the path of the saved preprocessing object.

        Example:
        >>> transformer = DataTransformation()
        >>> train_data, test_data, preprocessor_path = transformer.initiate_data_transformation('train.csv', 'test.csv')
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('Read train and test data into transformation process')
            
            preprocessor = self.get_data_transformer_object()
            logging.info('Obtained preprocessor object')

            # get the target, numerical & categorical column names
            target_column_name = 'math score'
            
            # split the train, test dataframes into train, test - feature, taget dataframes
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
            
            # fit & transform on feature dataframes -> this process returns feature arrays (instead of dataframe like input)
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info('Fit the preprocessing object with training data and used it to transform train & test data')
            
            # merge the transformed feature arrays & target arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
        except Exception as e:
            raise CustomException(e, sys)
        
        save_object(
            object=preprocessor,
            file_path=self.data_transformation_config.preprocessor_obj_file_path
        )
        
        return (
            train_arr, 
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path
            )
