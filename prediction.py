import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
model = pickle.load(open(ROOT/"model/ticket_classifier.pkl", "rb"))
vectorizer = pickle.load(open(ROOT/"model/tfidf_vectorizer.pkl", "rb"))

def predict_ticket(text):
    if not text.strip():
        return "Please enter a ticket."
    X = vectorizer.transform([text])
    return model.predict(X)[0]
