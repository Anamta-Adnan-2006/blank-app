import streamlit as st
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
# CUSTOM CSS
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
    font-size: 22px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# TITLE
# ===============================
st.markdown("<h1>🧠 Mental Health Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.write("✨ *AI-powered Mental Health Analysis System*")
st.divider()

# ===============================
# LOAD MODEL & TOOLS
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
# USER INPUT
# ===============================
st.subheader("💬 Enter a Statement")
user_text = st.text_area(
    "📝 Write your thoughts here:",
    placeholder="I feel very lonely and stressed these days...",
    height=150
)

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Analyze Mental Health"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter a statement first")
    else:
        cleaned = clean_text(user_text)
        vect = vectorizer.transform([cleaned])
        vect = scaler.transform(vect)

        prediction = model.predict(vect)[0]

        # ---------------------------
        # LABEL & COLOR MAPPING
        # ---------------------------
        if prediction.lower() == "normal":
            label = "🟢 Normal"
            color = "#c8f7c5"

        elif prediction.lower() in ["depressed", "anxious", "depression", "anxiety"]:
            label = "😔 Depressed / Anxious"
            color = "#ffeaa7"

        elif "suicidal" in prediction.lower():
            label = "🚨 Suicidal Ideation"
            color = "#fab1a0"

        elif "bipolar" in prediction.lower():
            label = "🔄 Bipolar Disorder"
            color = "#a29bfe"

        elif "stress" in prediction.lower():
            label = "😫 Stress"
            color = "#fdcb6e"

        else:
            label = "🎭 Personality Disorder"
            color = "#81ecec"

        # ---------------------------
        # RESULT DISPLAY
        # ---------------------------
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
