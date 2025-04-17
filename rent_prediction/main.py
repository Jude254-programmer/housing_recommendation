from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware

# ✅ Create FastAPI app instance before using it
app = FastAPI()

# ✅ Now you can safely add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model and encoding maps
model = joblib.load('rent_predictor_model.pkl')
room_type_encoding = joblib.load('room_type_encoding.pkl')
location_encoding = joblib.load('location_encoding.pkl')

# PostgreSQL Connection Details
engine = create_engine('postgresql://postgres:Derrick@localhost:5432/student_housing')

# Fetch unique room types from the database to dynamically create the encoding map
def get_encoding_map():
    query = """
        SELECT DISTINCT room_type
        FROM housing_data
    """
    df = pd.read_sql(query, engine)
    room_type_encoding = {room: idx for idx, room in enumerate(df['room_type'].unique())}
    return room_type_encoding

# Fetch encoding map from the dataset
ROOM_TYPE_ENCODING = get_encoding_map()


class RentPredictionRequest(BaseModel):
    room_type: str  # Now expecting a string for room type
    distance_from_campus_km: float
    security_score: int
    infrastructure_score: int
    water_electricity_reliability: int
    internet_availability: int


@app.post("/predict_rent_and_recommendations")
def predict_rent_and_recommendations(data: RentPredictionRequest):
    try:
        # Check if room_type is valid
        if data.room_type.lower() not in room_type_encoding:
            raise HTTPException(status_code=400, detail="Invalid room type.")

        # Encode room_type
        encoded_room_type = room_type_encoding[data.room_type.lower()]

        # Convert input data into a DataFrame for prediction
        input_data = pd.DataFrame([{
            "room_type_encoded": encoded_room_type,
            "location_encoded": 0,
            "distance_from_campus_km": data.distance_from_campus_km,
            "security_score": data.security_score,
            "infrastructure_score": data.infrastructure_score,
            "water_electricity_reliability": data.water_electricity_reliability,
            "internet_availability": data.internet_availability
        }])

        # Predict rent price
        rent_prediction = model.predict(input_data)[0]

        # Location recommendation
        query = text("""
            SELECT location, rent_price, distance_from_campus_km, location_encoded
            FROM housing_data
            WHERE distance_from_campus_km <= :distance_from_campus_km
              AND security_score >= :security_score
              AND infrastructure_score >= :infrastructure_score
            ORDER BY rent_price ASC
            LIMIT 5
        """)

        # Execute query with parameters for recommendations
        top_locations = pd.read_sql(query, engine, params={
            'distance_from_campus_km': data.distance_from_campus_km,
            'security_score': data.security_score,
            'infrastructure_score': data.infrastructure_score
        })

        return {
            "predicted_rent_price": rent_prediction,
            "top_5_locations": top_locations.to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
