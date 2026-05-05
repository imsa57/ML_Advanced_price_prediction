import os
import sys
from src.advanced_price_prediction.logger import logging
from src.advanced_price_prediction.exception import CustomException
from src.advanced_price_prediction.utils import read_data_from_database



from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    submission_data_path:str = os.path.join('artifacts','submission.csv')
    # raw_data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            train_data, test_data, submission_data= read_data_from_database()

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            train_data.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path),exist_ok=True)
            test_data.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.submission_data_path),exist_ok=True)
            submission_data.to_csv(self.data_ingestion_config.submission_data_path,index=False,header=True)

            logging.info("Data ingestion completed")
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
                self.data_ingestion_config.submission_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)