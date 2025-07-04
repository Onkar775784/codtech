# sentiment_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK data
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("train.csv")  # Ensure train.csv is in the same folder
print("First 5 rows of the dataset:")
print(df.head())

# Text cleaning function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r'http\S+|@\S+|#\S+|[^A-Za-z\s]', '', str(text))
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply text cleaning
df['clean_tweet'] = df['tweet'].apply(clean_text)

# Show word cloud
all_words = ' '.join(df['clean_tweet'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Tweets")
plt.show()

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_tweet']).toarray()
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Sample predictions
df['predicted'] = model.predict(X)
print("\nSample Predictions:")
print(df[['tweet', 'label', 'predicted']].sample(10))

# Overall accuracy
correct = (df['label'] == df['predicted']).sum()
total = len(df)
print(f"\nCorrect Predictions: {correct}/{total}")
print(f"Model Accuracy: {correct/total:.2f}")
