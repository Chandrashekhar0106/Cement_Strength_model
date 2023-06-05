import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)


class CustomData:
    def __init__(  self,
        cement: float,
        slag: float,
        flyash: float,
        water: float,
        superplasticizer: float,
        coarseaggregate: float,
        fineaggregate: float,
        age: int):

        self.cement = cement

        self.slag = slag

        self.flyash = flyash

        self.water = water

        self.superplasticizer = superplasticizer

        self.coarseaggregate = coarseaggregate

        self.fineaggregate = fineaggregate

        self.age = age

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "cement": [self.cement],
                "slag": [self.slag],
                "flyash": [self.flyash],
                "water": [self.water],
                "superplasticizer": [self.superplasticizer],
                "coarseaggregate": [self.coarseaggregate],
                "fineaggregate": [self.fineaggregate],
                "age": [self.age]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)