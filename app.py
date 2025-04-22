import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data
training_data = [
    {"intent": "greeting", "patterns": ["hello", "hi", "good morning"], "response": "Hi there! How can I help you?"},
    {"intent": "fever", "patterns": ["I have a fever", "feverish", "hot body"], "response": "Sounds like a fever. Stay hydrated and consult a doctor."},
    {"intent": "cold", "patterns": ["runny nose", "I have a cold"], "response": "It seems like a cold. Rest and drink warm fluids."},
    {"intent": "goodbye", "patterns": ["bye", "see you later"], "response": "Goodbye! Stay safe."}
]

# Preprocessing
corpus = []
labels = []
responses = {}

for data in training_data:
    for pattern in data["patterns"]:
        corpus.append(pattern.lower())
        labels.append(data["intent"])
    responses[data["intent"]] = data["response"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
model = LogisticRegression()
model.fit(X, labels)

# Chatbot response
def get_response(user_input):
    user_input = user_input.lower()
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    return responses.get(prediction, "Sorry, I didn't understand that. Can you rephrase?")

# Streamlit UI
st.title("🩺 Healthcare Chatbot")
st.write("This is a simple chatbot to help with basic symptom checking.")

user_input = st.text_input("You: ", "")

if user_input:
    response = get_response(user_input)
    st.text_area("Chatbot:", value=response, height=100)
