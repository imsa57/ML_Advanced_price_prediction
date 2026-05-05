import sys
from src.advanced_price_prediction.logger import logging
from src.advanced_price_prediction.exception import CustomException
from src.advanced_price_prediction.components.data_ingestion import DataIngestion
from src.advanced_price_prediction.components.data_transformation import DataTransformation
from src.advanced_price_prediction.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        train_data_path, test_data_path, submission_data_path = DataIngestion().initiate_data_ingestion()

        train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(train_data_path, test_data_path, submission_data_path)
        ModelTrainer().initiate_model_trainer(train_arr, test_arr)
    except Exception as e:
        raise CustomException(e,sys)