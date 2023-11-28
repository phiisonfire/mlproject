"""Module: data_ingestion

This module contains functions and classes for the data ingestion process in a machine learning project.

Functions:
    - load_data: Load data from various sources.
    - clean_data: Perform data cleaning and preprocessing.
    - explore_data: Conduct exploratory data analysis.
    - feature_engineering: Create and transform features for machine learning.

Classes:
    - DataProcessor: A class for handling the data processing tasks.

Usage:
    Example usage of the functions and classes can be found at the end of the file or in a separate script.

Author:
    Your Name <your.email@example.com>

"""
import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass # @dataclass helps generate special methods like __init__, __repr__ & __eq__ automatically based on its annotated attributes
class DataIngestionConfig:
    '''
    A data class for specifying paths to save output data.

    Attributes:
        train_data_path (str): The path to save training data. Defaults to 'artifacts/train.csv'.
        test_data_path (str): The path to save test data. Defaults to 'artifacts/test.csv'.
        raw_data_path (str): The path to save raw data. Defaults to 'artifacts/raw.csv'.

    Usage:
        config = DataIngestionConfig()
        print(config.train_data_path)  # Outputs 'artifacts/train.csv'
    '''
    train_data_path: str=os.path.join('artifacts', 'train.csv') # artifacts folder will store program outputs
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    '''
        A class for data ingestion, including reading data from a source, performing train-test split, 
        and saving raw, training, and testing datasets.

        Attributes:
            ingestion_config (DataIngestionConfig): An instance of DataIngestionConfig specifying paths for data storage.

        Usage:
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_date_ingestion()
    '''
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_date_ingestion(self):
        '''
        Read data from the source, perform train-test split, and save datasets.
        
        Returns:
            tuple: A tuple containing the paths to the training and testing datasets.
            
        Raises:
            CustomException: If an error occurs during the data ingestion process.
            
        Customizable:
            We can customize logic inside the first try-except block when the dataset is from another source but not csv file saved in local hard disk.
        '''
        logging.info('Entered the data ingestion method or component')
        
        # Try to load data from source into program
        try:
            df = pd.read_csv('notebook\StudentsPerformance.csv')
            logging.info('Read the dataset from notebook\StudentsPerformance.csv into memory as pandas.Dataframe')
        except Exception as e:
            raise CustomException(e, sys)
        
        # Try to split the data into train/test sets and save it in artifacts folder
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # create folder `artifacts`
            
            df.to_csv(path_or_buf=self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train Test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(path_or_buf=self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(path_or_buf=self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info('Ingestion of the data is completed')
        except Exception as e:
                    raise CustomException(e, sys)
        
        return (
            self.ingestion_config.train_data_path, 
            self.ingestion_config.test_data_path
        )
        
        
if __name__ == '__main__':
    data_ingestor = DataIngestion()
    data_ingestor.initiate_date_ingestion()

