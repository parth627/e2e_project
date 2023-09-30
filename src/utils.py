import os
import sys
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as exception:
        raise CustomException(exception, sys)
    
def evaluate_model(true_values, predicted_values):
    # mae = mean_absolute_error(true_values, predicted_values)
    # mse = mean_squared_error(true_values, predicted_values)
    # rmse = np.sqrt(mse)
    r_square_score = r2_score(true_values, predicted_values)
    # return mae,mse,rmse,r_square_score
    return r_square_score