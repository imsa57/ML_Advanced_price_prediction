import sys
import os
from dataclasses import dataclass
from src.advanced_price_prediction.exception import CustomException
from src.advanced_price_prediction.logger import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from src.advanced_price_prediction.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self,train_df):
        try:
            num_features = train_df.select_dtypes(exclude=['object']).columns
            cat_features = train_df.select_dtypes(include=['object']).columns

            num_pipeline = Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            cat_pipeline = Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder',OneHotEncoder(sparse_output=False)),
                ('scaler',StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,num_features),
                ('cat_pipeline',cat_pipeline,cat_features)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path,submission_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            submission_df = pd.read_csv(submission_path)

            X_train = train_df.drop('SalePrice',axis=1)
            y_train = train_df['SalePrice']
            X_test = test_df
            y_test = submission_df['SalePrice']

            preprocessing_obj = self.get_data_transformer_object(X_train)


            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr,np.array(y_train)]
            test_arr = np.c_[X_test_arr,np.array(y_test)]

            logging.info(f"DataTransformation Completed Successfully : [Train array shape: {train_arr.shape}], [Test array shape: {test_arr.shape}]")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
           
        except Exception as e:
            raise CustomException(e,sys)