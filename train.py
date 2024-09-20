import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load processed data
df = pd.read_csv('processed_data.csv')

# Define features (X) and target (y)
X = df.drop(['num_orders'], axis=1)  # Replace 'num_orders' with your actual target column
y = df['num_orders']

# Create the train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Initialize RandomForestRegressor with basic parameters
rf_model = RandomForestRegressor(n_jobs=-1,criterion='squared_error',random_state=101)

# Train the model
rf_model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(rf_model, 'model.joblib')

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r2}")

# Save evaluation metrics to a text file
try:
    with open('evaluation_results.txt', 'w') as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R-squared: {r2}\n")
    print("Evaluation metrics saved successfully to 'evaluation_results.txt'")
except Exception as e:
    print(f"Error writing evaluation results: {e}")
