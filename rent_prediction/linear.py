import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- PostgreSQL Connection Details ---
username = 'postgres'
password = 'Derrick'
host = 'localhost'
port = '5432'
database = 'student_housing'
table_name = 'housing_data'

# --- Create Engine and Load Data ---
engine = create_engine(f'postgresql://postgres:Derrick@localhost:5432/student_housing')
query = f"SELECT * FROM housing_data"
df = pd.read_sql(query, engine)

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

# --- Train Linear Regression Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Make Predictions ---
y_pred = model.predict(X_test)

# --- Evaluate the Model ---
r2 = r2_score(y_test, y_pred)
import numpy as np

# Calculate RMSE manually
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", r2)
print("RMSE:", rmse)
