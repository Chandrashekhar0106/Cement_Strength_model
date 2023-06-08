from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

#app=application

## Route for a home page

@application.route('/')
def index():
    return render_template('index.html') 

@application.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            cement=float(request.form.get('cement')),
            slag=float(request.form.get('slag')),
            flyash=float(request.form.get('flyash')),
            water=float(request.form.get('water')),
            superplasticizer=float(request.form.get('superplasticizer')),
            coarseaggregate=float(request.form.get('coarseaggregate')),
            fineaggregate=float(request.form.get('fineaggregate')),
            age=int(request.form.get('age'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    application.run(host="0.0.0.0", port=5000)        