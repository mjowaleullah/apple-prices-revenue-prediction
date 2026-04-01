from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib, numpy as np

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



model = joblib.load("model.pkl")

class apple_input(BaseModel):
    Country : float
    Apple_Variety: float
    Quality_Grade:float
    Market_Type: float
    Price_per_KG_USD: float
    Quantity_Sold_KG: float

@app.get("/")

def home():
    return {"status" : "Apple Prices API is Ready!"}

@app.post("/predict")

def predict(data: apple_input):
    features = np.array([[
        data.Country,
        data.Apple_Variety, 
        data.Quality_Grade,
        data.Market_Type,
        data.Price_per_KG_USD,
        data.Quantity_Sold_KG]])
    
    price = model.predict(features)[0]
    return {"predicted_price": round(float(price), 2)}