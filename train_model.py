import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Create Models directory if it doesn't exist
if not os.path.exists('Models'):
    os.makedirs('Models')

# Load dataset
df = pd.read_csv(r'F:\year2\MiniProject\reviews.csv')  # Ensure the path to the CSV file is correct

# Basic preprocessing
df['Reviews'] = df['Reviews'].str.replace('[^a-zA-Z]', ' ')
df['Reviews'] = df['Reviews'].str.lower()

# Define sentiment: 1 for positive (label >= 3), 0 for negative (label < 3)
df['sentiment'] = df['label'].apply(lambda x: 1 if x >= 3 else 0)

# Split dataset
X = df['Reviews']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
cv = CountVectorizer(stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# Scaling
scaler = StandardScaler(with_mean=False)
X_train_scl = scaler.fit_transform(X_train_cv)
X_test_scl = scaler.transform(X_test_cv)

# Model training
model = XGBClassifier()
model.fit(X_train_scl, y_train)

# Save the model and transformers
with open('Models/model_xgb.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('Models/countVectorizer.pkl', 'wb') as file:
    pickle.dump(cv, file)
with open('Models/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Model trained and saved successfully.")

# Predictions on the test set
y_pred = model.predict(X_test_scl)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate classification report
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
