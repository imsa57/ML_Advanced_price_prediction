import os
import sys
from src.advanced_price_prediction.logger import logging
from src.advanced_price_prediction.exception import CustomException
from sklearn.model_selection import GridSearchCV
import pandas as pd
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
# C:\Users\SARFRAJ\Desktop\End-to-End data science Project\End-to-End-Advanced-price-prediction\dataset
path = os.path.join("C:\\Users","SARFRAJ","Desktop","End-to-End data science Project","End-to-End-Advanced-price-prediction","dataset")
def read_data_from_database():
    try:
        train_data = pd.read_csv(path+'/train.csv')
        test_data = pd.read_csv(path+'/test.csv')
        submission_data = pd.read_csv(path+'/submission.csv')
        logging.info(f"Data reading completed successfully with shape: [train_data: {train_data.shape}], [test_data: {test_data.shape}], [submission_data: {submission_data.shape}]")
        return train_data, test_data, submission_data
    except Exception as e:
        raise CustomException(e,sys)


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=10)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_r2_score = r2_score(y_train,y_train_pred)
            test_model_r2_score = r2_score(y_test,y_test_pred)

            train_model_mae = mean_absolute_error(y_train,y_train_pred)
            test_model_mae = mean_absolute_error(y_test,y_test_pred)
            
            object_report = {
                'train_model_r2_score': train_model_r2_score,
                'test_model_r2_score': test_model_r2_score,
                'train_model_mae': train_model_mae,
                'test_model_mae': test_model_mae,
                'best_params': gs.best_params_,
                'best_model': list(models.keys())[i],
            }


            report[list(models.keys())[i]] = object_report
        return report
    except Exception as e:
        raise CustomException(e,sys)