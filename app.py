import streamlit as st
import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")

# Load intents.json
with open("intents.json") as file:
    data = json.load(file)

# Prepare data
X = []
y = []
responses = {}

for intent in data["intents"]:
    tag = intent["tag"]
    responses[tag] = intent["responses"]
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(tag)

# Create pipeline (TF-IDF + Logistic Regression)
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(X, y)

# Streamlit UI
st.title("🩺 Medical Chatbot")
st.markdown("Ask me anything about your health concerns.")

user_input = st.text_input("You:", "")

if user_input:
    prediction = model.predict([user_input])[0]
    bot_response = random.choice(responses[prediction])
    st.markdown(f"🤖: {bot_response}")
