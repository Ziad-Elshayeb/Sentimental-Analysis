import streamlit as st
import joblib
from utils import predict_sentiment
import nltk
nltk.download('punkt')

model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit app interface
st.title("Sentiment Analysis")

st.header("Enter a Sentence to Analyze Sentiment")
input_text = st.text_area("Type your text here:")

if st.button("Analyze Sentiment"):
    if input_text.strip():
        prediction = predict_sentiment(input_text, model, vectorizer)
        st.write(f"Predicted Sentiment: **{prediction}**")
    else:
        st.write("Please enter some text to analyze!")
