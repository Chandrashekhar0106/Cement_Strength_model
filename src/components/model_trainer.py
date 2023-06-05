import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Random Forest": RandomForestRegressor(bootstrap=False),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Random Forest":{
                    "n_estimators": [100, 200, 500, 600, 800, 900],
                    "criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    "max_depth": [None, 2, 3, 5, 7, 8],
                    "min_samples_split": [2, 5, 7, 9, 10],
                    "min_samples_leaf": [1, 2, 5, 7, 9, 10],
                    "max_features": ["auto", "sqrt", "log2"]
                },
                "Gradient Boosting":{
                    "n_estimators": [100, 200, 500, 600, 800, 900],
                    "criterion": ['friedman_mse', 'squared_error'],
                    "learning_rate": [0.05,0.10,0.15,0.20,0.25,0.30],
                    "max_depth": [None, 2, 3, 5, 7, 8],
                    "min_samples_split": [2, 5, 7, 9, 10],
                    "min_samples_leaf": [1, 2, 5, 7, 9, 10]
                },
                "Linear Regression":{},
                "Lasso":{},
                "Ridge":{},
                "K-Neighbors Regressor":{},
                "Decision Tree":{},
                "AdaBoost Regressor":{},
                "XGBRegressor":{
                    'tree_method': ['exact','hist','gpu_hist','approx'],
                    'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
                    'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
                    'min_child_weight' : [ 1, 3, 5, 7 ],
                    'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
                    'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ],
                    'n_estimators':[300,400,500,600,700,800,900]
                },
                "CatBoosting Regressor":{
                    'depth'          : [4,5,6,7,8,9, 10],
                    'learning_rate' : [0.01,0.02,0.03,0.04],
                    'iterations'    : [300,400,500,600,700,800,900]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)