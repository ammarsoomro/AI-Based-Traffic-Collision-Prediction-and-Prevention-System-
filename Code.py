import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

np.random.seed(42)
data_size = 1000

data = pd.DataFrame({
    'speed_kmph': np.random.randint(20, 120, size=data_size),
    'weather_condition': np.random.choice(['clear', 'rain', 'fog', 'snow'], size=data_size),
    'traffic_density': np.random.choice(['low', 'medium', 'high'], size=data_size),
    'driver_alert': np.random.choice([0, 1], size=data_size),
    'collision_risk': np.random.choice([0, 1], size=data_size)
})

data = pd.get_dummies(data, columns=['weather_condition', 'traffic_density'], drop_first=True)

X = data.drop('collision_risk', axis=1)
y = data['collision_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

def predict_collision(speed, weather, traffic, alert):
    input_df = pd.DataFrame([{
        'speed_kmph': speed,
        'driver_alert': alert,
        'weather_condition_clear': 1 if weather == 'clear' else 0,
        'weather_condition_fog': 1 if weather == 'fog' else 0,
        'weather_condition_rain': 1 if weather == 'rain' else 0,
        'weather_condition_snow': 1 if weather == 'snow' else 0,
        'traffic_density_low': 1 if traffic == 'low' else 0,
        'traffic_density_medium': 1 if traffic == 'medium' else 0,
        'traffic_density_high': 1 if traffic == 'high' else 0
    }])

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]

    result = model.predict(input_df)[0]
    return "High Collision Risk" if result == 1 else "Low Collision Risk"

print(predict_collision(speed=85, weather='rain', traffic='high', alert=0))
