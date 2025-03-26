import psycopg2
import pandas as pd

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="student_housing",
    user="******",
    password="******",
    host="localhost",
    port="5432"
)

# Load data
query = "SELECT * FROM housing_data;"
df = pd.read_sql(query, conn)
conn.close()

# Display data
print(df.head())

student_weights = {
    "rent": 0.3,  # Higher value → more importance on affordability
    "distance": 0.2,  # Higher value → prefers houses closer to campus
    "infrastructure": 0.25,  # Higher value → better amenities matter
    "security": 0.25  # Higher value → prioritizes security
}
# Normalize values (scale to 0-1)
df['RENT_SCORE'] = 1 / (df['rent_price'] + 1)  # Lower rent = higher score
df['DISTANCE_SCORE'] = 1 / (df['distance_from_campus_km'] + 1)  # Closer = better
df['INFRASTRUCTURE_SCORE_NORM'] = df['infrastructure_score'] / 10  # Convert to 0-1 scale
df['SECURITY_SCORE_NORM'] = df['security_score'] / 10  # Convert to 0-1 scale

# Compute recommendation score based on student preferences
df['RECOMMENDATION_SCORE'] = (
    student_weights["rent"] * df['RENT_SCORE'] +
    student_weights["distance"] * df['DISTANCE_SCORE'] +
    student_weights["infrastructure"] * df['INFRASTRUCTURE_SCORE_NORM'] +
    student_weights["security"] * df['SECURITY_SCORE_NORM']
)

# Show results
print(df[['location', 'rent_price', 'distance_from_campus_km', 'RECOMMENDATION_SCORE']].sort_values(by='RECOMMENDATION_SCORE', ascending=False).head())

top_recommendations = df[['location', 'RECOMMENDATION_SCORE']].sort_values(by='RECOMMENDATION_SCORE', ascending=False).head(5)
print("Top Recommended Locations:")
print(top_recommendations)

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop(columns=['RECOMMENDATION_SCORE'])  # Drop the target column
y = df['RECOMMENDATION_SCORE']  # Use recommendation score as target

# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split complete!")
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

#Model train using Random Forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize model
model = RandomForestRegressor(n_estimators=100, random_state=42)


# Drop non-numeric columns
X_train = X_train.drop(columns=['location', 'room_type'])
X_test = X_test.drop(columns=['location', 'room_type'])
#print(X_train.dtypes)


# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

#import matplotlib.pyplot as plt
#import numpy as np

# Get feature importance
#feature_importance = model.feature_importances_
#feature_names = X_train.columns

# Plot feature importance
#plt.figure(figsize=(10, 5))
#plt.barh(feature_names, feature_importance, color="skyblue")
#plt.xlabel("Feature Importance Score")
#plt.ylabel("Features")
#plt.title("Feature Importance in Recommendation Model")
#plt.show()
#import matplotlib.pyplot as plt

#plt.figure(figsize=(8, 6))
#plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
#plt.plot(y_test, y_test, color="red", linestyle="--")  # Perfect predictions line
#plt.xlabel("Actual Values")
#plt.ylabel("Predicted Values")
#plt.title("Actual vs. Predicted Recommendations")
#plt.show()


#Testing the model
import pandas as pd
import numpy as np

# Sample Student Preferences
student_preferences = {
    'room_type_encoded': 0,  # Example encoding for "Single"
    'distance_from_campus_km': 2.0,  # Preferred max distance
    'rent_price': 7000,  # Maximum budget
    'infrastructure_score': 7,  # Infrastructure importance
    'security_score': 10,  # Security preference
    'water_electricity_reliability': 9,  # Utility reliability
    'internet_availability': 7  # Need for internet access
}

# Convert preferences into a DataFrame
df_student = pd.DataFrame([student_preferences])

# Normalize the preference values as done during training
df_student['RENT_SCORE'] = df_student['rent_price'] / df['rent_price'].max()
df_student['DISTANCE_SCORE'] = df_student['distance_from_campus_km'] / df['distance_from_campus_km'].max()
df_student['INFRASTRUCTURE_SCORE_NORM'] = df_student['infrastructure_score'] / 10  # Assuming scores out of 10
df_student['SECURITY_SCORE_NORM'] = df_student['security_score'] / 10  # Assuming scores out of 10

# Ensure features match model training data
df_student = df_student.reindex(columns=X_train.columns, fill_value=0)


# Filter available houses that are within budget
affordable_houses = df[df['rent_price'] <= student_preferences['rent_price']]

# Check if there are any affordable houses
if affordable_houses.empty:
    print("No available houses within budget.")
else:
    # Predict scores for affordable housing locations
    affordable_houses['predicted_score'] = model.predict(affordable_houses[X_train.columns])

    # Find the best affordable location based on the highest predicted score
    best_location = affordable_houses.loc[
        affordable_houses['predicted_score'].idxmax(), ['location', 'rent_price', 'predicted_score']]

    print(f"🏠 Recommended Location: {best_location['location']}")
    print(f"💰 Rent Price: Ksh {best_location['rent_price']}")
    print(f"📊 Predicted Score: {best_location['predicted_score']:.4f}")



