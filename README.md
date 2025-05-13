# 💉 Medical Chatbot with Streamlit

A simple and interactive **Medical Chatbot** built using **Python**, **Streamlit**, and **Machine Learning**.  
It uses a custom intents dataset (`intents1.json`) to classify user queries related to **healthcare, symptoms, or medicines**, and provides intelligent responses.

---

## 🧠 Features

- Trained using **TF-IDF + Random Forest Classifier**
- Handles multiple intent-based medical questions
- Responds with varied answers for each intent
- Easy-to-use **web interface** via Streamlit

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- JSON for intent data
- TfidfVectorizer + RandomForestClassifier

---

## 🧪 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot
pip install -r requirements.txt
streamlit run chatbot_app.py

