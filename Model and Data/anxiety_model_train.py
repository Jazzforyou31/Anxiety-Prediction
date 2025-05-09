import pickle
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load dataset
print("Loading dataset...")
dataset = pd.read_csv('train.csv')
dataset.head()

# Clean column names (strip spaces and convert to lowercase)
dataset = dataset.rename(columns=lambda x: x.strip().lower())

# Display basic info
print(f"Dataset shape: {dataset.shape}")
print("Feature names:", ', '.join(dataset.columns[:-1]))
print("Target variable:", dataset.columns[-1])

# Extract features and target
X = dataset.drop(['anxiety score'], axis=1)
y = dataset['anxiety score']

# Handle any missing values (if any)
for col in X.columns:
    if X[col].isnull().sum() > 0:
        if X[col].dtype.kind in 'ifc':  # integer, float, complex
            X[col] = X[col].fillna(X[col].mean())
        else:
            X[col] = X[col].fillna(X[col].mode()[0])

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Define the model
print("Training Gradient Boosting model...")
anxiety_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.9,
    random_state=42
)

# Train the model
anxiety_model.fit(X_train, y_train)

# Evaluate the model
y_pred = anxiety_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Create a dictionary with feature importance
feature_importance = dict(zip(X.columns, anxiety_model.feature_importances_))
sorted_features = dict(sorted(feature_importance.items(), 
                             key=lambda x: x[1], 
                             reverse=True))

print("\nFeature Importance:")
for feature, importance in list(sorted_features.items())[:5]:
    print(f"{feature}: {importance:.4f}")

# Save model package to disk
print("\nSaving model and scaler to disk...")
model_package = {
    'model': anxiety_model,
    'feature_names': list(X.columns),
    'scaler': scaler,
    'metrics': {
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    },
    'feature_importance': feature_importance,
    'training_date': pd.Timestamp.now()
}

# Create directory if it doesn't exist
anxiety_dir = 'anxiety_model'
if not os.path.exists(anxiety_dir):
    os.makedirs(anxiety_dir)

# Save model package to both current directory and anxiety_model directory
pickle.dump(model_package, open("anxiety_prediction_model.pkl", "wb"))
pickle.dump(model_package, open(os.path.join(anxiety_dir, "anxiety_prediction_model.pkl"), "wb"))

# Also save individual components
pickle.dump(anxiety_model, open(os.path.join(anxiety_dir, "gb_model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(anxiety_dir, "scaler.pkl"), "wb"))

print("Model files saved successfully to:")
print(f"1. Current directory: {os.path.abspath('anxiety_prediction_model.pkl')}")
print(f"2. Anxiety model directory: {os.path.abspath(anxiety_dir)}")

# Save a sample prediction script
prediction_script = """
# Example usage of the anxiety prediction model
import pickle
import pandas as pd
import numpy as np

def load_model(model_path='anxiety_prediction_model.pkl'):
    # Load the model package
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    # Extract components
    model = model_package['model']
    feature_names = model_package['feature_names']
    scaler = model_package['scaler']
    
    return model, feature_names, scaler

def predict_anxiety(input_data, model, feature_names, scaler):
    # Ensure input data has all required features
    missing_features = [f for f in feature_names if f not in input_data.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")
    
    # Select only the required features in the correct order
    X = input_data[feature_names].copy()
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return predictions

# Example of how to use the model
if __name__ == "__main__":
    # Load the model
    model, feature_names, scaler = load_model()
    
    # Create sample input data (replace with actual data)
    sample_data = {
        'stress_sleep_impact': 7.5,
        'stress_impact_score': 85.2,
        'sleep_stress_severity': 8.1,
        'sleep hours': 5.5,
        'activity_stress_balance': -2.3,
        'high_risk_factors': 3,
        'therapy sessions (per month)': 1,
        'stress_sleep_composite': 6.8,
        'health_risk_score': 72.5
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([sample_data])
    
    # Make prediction
    predicted_anxiety = predict_anxiety(input_df, model, feature_names, scaler)
    
    print(f"Predicted Anxiety Score: {predicted_anxiety[0]:.2f}")
"""

with open(os.path.join(anxiety_dir, "prediction_example.py"), "w") as f:
    f.write(prediction_script)

print("\nExample prediction script saved to anxiety_model directory.")
print("Training complete!") 