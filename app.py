import streamlit as st
import pandas as pd
import re
import string
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="🧠 Mental Health Analyzer",
    page_icon="💙",
    layout="centered"
)

# ===============================
# CUSTOM CSS (Colorful UI)
# ===============================
st.markdown("""
<style>
body {
    background-color: #f4f9ff;
}
.main {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 20px;
}
h1 {
    color: #4b6cb7;
}
.pred-box {
    padding: 20px;
    border-radius: 15px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# TITLE
# ===============================
st.markdown("<h1>🧠 Mental Health Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.write("✨ *AI powered mental health analysis system*")

st.divider()

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================
model = joblib.load("mental_health_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# TEXT CLEANING FUNCTION
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text

# ===============================
# CLASS MAPPING
# ===============================
class_labels = {
    0: "🟢 Normal",
    1: "😔 Depressed / Anxious",
    2: "🚨 Suicidal Ideation",
    3: "🔄 Bipolar Disorder",
    4: "😫 Stress",
    5: "🎭 Personality Disorder"
}

class_colors = {
    0: "#c8f7c5",
    1: "#ffeaa7",
    2: "#fab1a0",
    3: "#a29bfe",
    4: "#fdcb6e",
    5: "#81ecec"
}

# ===============================
# USER INPUT
# ===============================
st.subheader("💬 Enter a statement")
user_text = st.text_area(
    "📝 Write your thoughts here:",
    placeholder="I feel very lonely and stressed these days...",
    height=150
)

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("🔍 Analyze Mental Health"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter a statement first")
    else:
        cleaned = clean_text(user_text)
        vect = vectorizer.transform([cleaned])
        vect = scaler.transform(vect)

        prediction = model.predict(vect)[0]
        label = class_labels[prediction]
        color = class_colors[prediction]

        st.markdown(
            f"""
            <div class="pred-box" style="background-color:{color}">
                💡 Prediction Result <br><br>
                {label}
            </div>
            """,
            unsafe_allow_html=True
        )

# ===============================
# FOOTER
# ===============================
st.divider()
st.markdown("💙 **AI for Mental Health Awareness**")
st.markdown("⚠️ *This tool is for educational purposes only*")