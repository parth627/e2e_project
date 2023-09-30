import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException 
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("../artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing input")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regressor": LinearRegression(),
                "Random Forest Regressor":RandomForestRegressor(),
                "Gradient Boosting Regressor":GradientBoostingRegressor(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Cat Boosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoostRegressor": AdaBoostRegressor()
            }

            model_report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train,y_train)

                y_test_pred = model.predict(X_test)
                r_square_score = r2_score(y_test,y_test_pred)

                model_report[list(models)[i]] = r_square_score
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report)[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found for training and testing data")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test,predicted)

            return r2_square
        
        except Exception as exception:
            raise CustomException(exception,sys)       
        
