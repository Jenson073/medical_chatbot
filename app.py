import streamlit as st
import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# Load the intents file
with open("intents1.json", "r") as file:
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

# Create the pipeline
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100))
model.fit(X, y)

# Streamlit UI
st.title("💉 Medical Chatbot")
st.markdown("Feel free to ask anything related to healthcare, symptoms, or medicines.")

user_input = st.text_input("You:", "")

if user_input:
    prediction = model.predict([user_input])[0]
    response = random.choice(responses[prediction])
    st.markdown(f"🤖: {response}")
