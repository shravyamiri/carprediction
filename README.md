import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

car_data = pd.read_csv('/content/car data.csv')

print(car_data.head())
print(car_data.isnull().sum())

X = car_data[['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']]
y = car_data['Selling_Price']
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

missing_features = set(X_train.columns) - set(X_test.columns)
for feature in missing_features:
    X_test[feature] = 0

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

print("Training Feature Names:")
print(X_train.columns)

# Print the order of feature names for testing dataset
print("\nTesting Feature Names:")
print(X_test.columns)

custom_data = pd.DataFrame({
    'Year': [2018],
    'Present_Price': [10.0],
    'Driven_kms': [50000],
    'Owner': [0],
    'Fuel_Type_CNG': [0],
    'Fuel_Type_Diesel': [0],
    'Fuel_Type_Petrol': [1],
    'Selling_type_Dealer': [0],
    'Selling_type_Individual': [1],
    'Transmission_Automatic': [0],
    'Transmission_Manual': [1]
})

custom_prediction = rf_regressor.predict(custom_data)

print(f"Predicted Selling Price for Custom Data: {custom_prediction[0]:.2f}")

