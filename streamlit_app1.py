import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Title
st.title("Course Reviews Sentiment Analysis")

# Upload CSV data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the data
    st.write(df.head())

    # Load the pre-trained model and vectorizer
    model = joblib.load('Models/model_xgb.pkl')
    vectorizer = joblib.load('Models/tfidf_vectorizer.pkl')
    scaler = joblib.load('Models/scaler.pkl')

    # Transform the reviews
    reviews_vectorized = vectorizer.transform(df['Reviews'])
    reviews_scaled = scaler.transform(reviews_vectorized)

    # Predict sentiments in chunks to avoid memory issues
    chunk_size = 1000  # Adjust the chunk size based on available memory
    predictions = []

    for start in range(0, reviews_scaled.shape[0], chunk_size):
        end = min(start + chunk_size, reviews_scaled.shape[0])
        reviews_chunk = reviews_scaled[start:end]
        chunk_predictions = model.predict(reviews_chunk)
        predictions.extend(chunk_predictions)

    df['predicted_sentiment'] = predictions

    # Display the predictions
    st.write(df[['Reviews', 'predicted_sentiment']])
else:
    st.write("Please upload a CSV file containing the reviews.")
