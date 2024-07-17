import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

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
import pickle
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

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the Confusion Matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Pie chart for sentiment distribution in the test set
sentiment_counts = y_test.value_counts()
labels = ['Positive', 'Negative']
sizes = [sentiment_counts[1], sentiment_counts[0]]
colors = ['#ff9999','#66b3ff']
explode = (0.1, 0)  # explode 1st slice (Positive)

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Sentiment Distribution in the Test Set')
plt.show()

# Bar chart for the precision, recall, and f1-score for each class
classification_report_data = {
    'class': ['Negative', 'Positive'],
    'precision': [0.74, 0.97],
    'recall': [0.26, 1.00],
    'f1-score': [0.38, 0.98]
}

df_classification_report = pd.DataFrame(classification_report_data)

plt.figure(figsize=(10, 6))
bar_width = 0.2
index = range(len(df_classification_report))

plt.bar(index, df_classification_report['precision'], bar_width, label='Precision', color='b')
plt.bar([i + bar_width for i in index], df_classification_report['recall'], bar_width, label='Recall', color='g')
plt.bar([i + 2 * bar_width for i in index], df_classification_report['f1-score'], bar_width, label='F1-Score', color='r')

plt.xlabel('Class')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1-Score by Class')
plt.xticks([i + bar_width for i in index], df_classification_report['class'])
plt.legend()
plt.show()

# Bar chart for feature importance
feature_importance_df = pd.DataFrame({'Feature': cv.get_feature_names_out(), 'Importance': model.feature_importances_})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Top 20 Important Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
