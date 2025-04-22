import streamlit as st
import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the intents JSON file
with open("intents.json", "r") as file:
    data = json.load(file)

# Extract patterns and tags
corpus = []
labels = []
responses = {}

for intent in data["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        corpus.append(pattern)
        labels.append(tag)
    responses[tag] = intent["responses"]

# Load Sentence Transformer model
st.session_state.embedder = st.session_state.get("embedder", SentenceTransformer("all-MiniLM-L6-v2"))
embedder = st.session_state.embedder

# Embed the entire corpus once
if "corpus_embeddings" not in st.session_state:
    st.session_state.corpus_embeddings = embedder.encode(corpus)

corpus_embeddings = st.session_state.corpus_embeddings

# Function to predict intent
def predict_intent(user_input):
    user_embedding = embedder.encode([user_input])
    similarities = cosine_similarity(user_embedding, corpus_embeddings)
    best_match_index = np.argmax(similarities)
    predicted_tag = labels[best_match_index]
    return predicted_tag

# Streamlit UI
st.title("💊 Medical Chatbot")
st.markdown("Ask me anything about symptoms, medicines, or health concerns!")

user_input = st.text_input("You:", "")

if user_input:
    intent = predict_intent(user_input)
    response = random.choice(responses[intent])
    st.markdown(f"🤖: {response}")
