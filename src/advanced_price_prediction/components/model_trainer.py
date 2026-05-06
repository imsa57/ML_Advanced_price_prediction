import os
import sys
import mlflow
from dataclasses import dataclass
from src.advanced_price_prediction.exception import CustomException
from src.advanced_price_prediction.logger import logging
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from src.advanced_price_prediction.utils import evaluate_models
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from src.advanced_price_prediction.utils import save_object
import dagshub
dagshub.init(repo_owner='imsa57', repo_name='ML_Advanced_price_prediction', mlflow=True)
# https://dagshub.com/imsa57/ML_Advanced_price_prediction.mlflow
mlflow.set_tracking_uri(f"https://dagshub.com/imsa57/ML_Advanced_price_prediction.mlflow")
mlflow.set_experiment("Advanced Price Prediction")


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    trained_model_report_file_path = os.path.join('artifacts','model_report.json')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
      try:
        X_train, y_train, X_test, y_test = (
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1]
        )
        alphas=np.arange(1,1000,100).tolist()
        print(alphas)

        models={
            'Linear Regression': LinearRegression(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'ElasticNet': ElasticNet(),
        }
        params={
            'Linear Regression': {},
            'Lasso': {
                'alpha': alphas,
                'max_iter': [100000],
            },
            'Ridge': {
                'alpha': alphas,
                'max_iter': [100000],
            },
            'ElasticNet': {
                'alpha': alphas,
                'max_iter': [100000],
            },
        }
        model_report = evaluate_models(X_train,y_train,X_test,y_test,models,params)

        with open(self.model_trainer_config.trained_model_report_file_path,'w') as f:
            json.dump(model_report,f,indent=4)

        best_model_data = min(model_report.values(),key=lambda x: x["test_model_mae"])
        best_model = best_model_data["best_model"]
        best_params = best_model_data["best_params"]
        logging.info(f"Best Parameters founded using Grid Search CV: Model: {best_model}, Params: {best_params} MAE: {best_model_data['test_model_mae']}")

        best_model = models[best_model]
        # best_model.set_params(**best_params)
        # best_model.fit(X_train,y_train)
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test,y_pred)
        test_r2_score = r2_score(y_test,y_pred)
        print(f"Best Model: {best_model}, Best Params: {best_params}, MAE: {mae}, R2 Score: {test_r2_score}")
        with mlflow.start_run():
            mlflow.sklearn.log_model(sk_model=best_model, name="best_model")
            mlflow.log_params(best_params)
            mlflow.log_metric('test_model_mae', mae)
            mlflow.log_metric('test_model_r2_score', test_r2_score)
            
        save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
        return mae,test_r2_score
      except Exception as e:
        raise CustomException(e,sys)