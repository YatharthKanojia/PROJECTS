import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load your dataset
df = pd.read_csv(r'F:\year2\MiniProject\reviews.csv')  # Change this to your dataset path

# Assuming 'review' column contains the text data
reviews = df['Reviews']

# Create TF-IDF Vectorizer and fit it
vectorizer = TfidfVectorizer(max_features=29722)
reviews_vectorized = vectorizer.fit_transform(reviews)

# Create and fit scaler
scaler = StandardScaler(with_mean=False)
reviews_scaled = scaler.fit_transform(reviews_vectorized)

# Ensure the Models directory exists
os.makedirs('Models', exist_ok=True)

# Save the vectorizer
joblib.dump(vectorizer, 'Models/tfidf_vectorizer.pkl')

# Save the scaler
joblib.dump(scaler, 'Models/scaler.pkl')

print("Vectorizer and scaler have been saved successfully.")
