# ==========================================
# Streamlit App: Mental Health Sentiment Analyzer
# ==========================================
import streamlit as st
import pandas as pd
import re
import string
import joblib
from sklearn.preprocessing import MaxAbsScaler

# ------------------------------
# Load saved model and vectorizer
# ------------------------------
model = joblib.load("mental_health_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------
# Helper function to clean text
# ------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

# ------------------------------
# Streamlit App UI
# ------------------------------
st.set_page_config(page_title="Mental Health Sentiment Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 Mental Health Sentiment Analyzer")
st.write("Type your thoughts or feelings below, and the app will predict your sentiment!")

# User input
user_input = st.text_area("Enter your sentence here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        vect = scaler.transform(vect)
        prediction = model.predict(vect)[0]

        # Emoji and color coding
        if prediction.lower() == "positive":
            sentiment = "Positive 😊"
            color = "#2ECC71"  # green
        elif prediction.lower() == "negative":
            sentiment = "Negative 😔"
            color = "#E74C3C"  # red
        else:
            sentiment = "Neutral 😐"
            color = "#F1C40F"  # yellow

        st.markdown(f"<h2 style='color:{color};'>Predicted Sentiment: {sentiment}</h2>", unsafe_allow_html=True)
        
        # Optional advice
        if prediction.lower() == "negative":
            st.info("💡 Take a deep breath, maybe talk to a friend or professional. You are not alone!")
        elif prediction.lower() == "positive":
            st.success("👍 Keep up the positive vibes!")
        else:
            st.warning("⚖️ Neutral mood detected. Stay mindful of your feelings.")
    else:
        st.error("Please enter some text to analyze.")
