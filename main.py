import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError


model = joblib.load("./model/pipeline.joblib")

app = FastAPI()

class DataPredict(BaseModel):
    data_to_predict: list[list] = [[]]
    

@app.post("/predict")
def predict(pred: list[list]):
    try:
        data = pred
        cols = ['Gender', 'Ethnicity', 'ParentalEducation', 'StudyTimeWeekly',
        'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports',
        'Music', 'Volunteering']
        df_data = pd.DataFrame(data, columns=cols)
        
        prediction = model.predict(df_data)
        return {"prediction": prediction.tolist()}
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

app.get("/")
def home():
    return {'Lugar de predicciones(TEST)'}