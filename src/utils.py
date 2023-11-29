import os
import sys

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(object, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(object, f)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models:dict) -> dict:
    try:
        model_performance = {}
        for model_name in models.keys():
            model = models[model_name]
            model.fit(x_train, y_train)
            y_test_pred = model.predict(x_test)
            r2 = r2_score(y_true=y_test, y_pred=y_test_pred)
            model_performance[model_name] = r2
    except Exception as e:
        raise CustomException(e, sys)
    
    return model_performance
        
        
    
    
    