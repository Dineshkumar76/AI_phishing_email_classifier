
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import string
import pickle

nltk.download('stopwords')

# Load pre-trained model and vectorizer
with open("phishing_email_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Function to clean email text
def clean_text(text):
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word.lower() for word in tokens if word.lower() not in stopwords.words('english')]
    return " ".join(tokens)

# Function to classify new emails
def classify_email(email_content):
    cleaned_email = clean_text(email_content)
    vectorized_email = vectorizer.transform([cleaned_email])
    prediction = model.predict(vectorized_email)
    return "Phishing" if prediction[0] == 1 else "Legitimate"

# Example usage
if __name__ == "__main__":
    test_email = "Congratulations! You've won a prize. Click here to claim it."
    result = classify_email(test_email)
    print(f"The email is classified as: {result}")
