import csv, pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = list(csv.DictReader(open("dataset/tickets.csv", encoding="utf-8")))
texts = [r["ticket_text"] for r in data]
labels = [r["category"] for r in data]

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X = vectorizer.fit_transform(texts)
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

Path("model").mkdir(exist_ok=True)
pickle.dump(model, open("model/ticket_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl", "wb"))
print("Model trained and saved successfully.")
