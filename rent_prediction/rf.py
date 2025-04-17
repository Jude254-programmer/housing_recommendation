import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- PostgreSQL Connection Details ---
username = 'postgres'
password = 'Derrick'
host = 'localhost'
port = '5432'
database = 'student_housing'
table_name = 'housing_data'

# --- Create Engine and Load Data ---
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')
query = f"SELECT * FROM housing_data"
df = pd.read_sql(query, engine)

# --- Encode room_type and location columns ---
# Room type encoding
room_type_encoding = {'single': 1, 'bedsitter': 0}  # Example encoding for room_type
df['room_type_encoded'] = df['room_type'].apply(lambda x: room_type_encoding.get(x.lower(), -1))  # Encoding room type (handling lowercase)

# Location encoding using hash function or a custom encoding
location_encoding = {location: idx for idx, location in enumerate(df['location'].unique())}  # Example encoding for location
df['location_encoded'] = df['location'].apply(lambda x: location_encoding.get(x.lower(), -1))  # Encoding location (handling lowercase)

# Save encoding maps to files for future use
joblib.dump(room_type_encoding, 'room_type_encoding.pkl')
joblib.dump(location_encoding, 'location_encoding.pkl')

# --- Select Features and Target ---
features = [
    'room_type_encoded',
    'location_encoded',
    'distance_from_campus_km',
    'security_score',
    'infrastructure_score',
    'water_electricity_reliability',
    'internet_availability'
]
target = 'rent_price'

X = df[features]
y = df[target]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Random Forest Model ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# --- Make Predictions ---
y_pred_rf = rf_model.predict(X_test)

# --- Evaluate the Model ---
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("Random Forest R² Score:", r2_rf)
print("Random Forest RMSE:", rmse_rf)

# --- Save the trained model ---
joblib.dump(rf_model, 'rent_predictor_model.pkl')
print("✅ Model saved as rent_predictor_model.pkl")
