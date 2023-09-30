'''
 This module help in creating required class and methods for data ingestion part.
'''
from dataclasses import dataclass
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split 
from src.exception import CustomException
from src.logger import logging
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

from src.components.data_transformation import DataTransformation,DataTransformationConfig

@dataclass
class DataIngestionConfig:
    '''
     This class is to define the config class for data ingestion.
    '''
    train_data_path:str = os.path.join('../artifacts','train.csv')
    test_data_path:str = os.path.join('../artifacts','test.csv')
    raw_data_path:str = os.path.join('../artifacts','raw.csv')

class DataIngestion:
    '''
    DataIngestion logic code
    '''
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
  
    def initiate_data_ingestion(self):
        '''
        This method is to start the data ingestion from local/cloud mongodb etc
        '''
        logging.info("Entered the data ingestion component")
        try:
            data_frame = pd.read_csv(r"C:\Users\parth\Desktop\DSML ipynbs\e2e_project\notebook\data\stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            # exist_ok=True -> if it is already there, we dont delete and create another 

            data_frame.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(data_frame,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)
        
        except Exception as exception:
            raise CustomException(exception,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array,test_array = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    model_trainer = ModelTrainer()
    final_score = model_trainer.initiate_model_trainer(train_array=train_array,test_array=test_array)

    print(final_score)
    