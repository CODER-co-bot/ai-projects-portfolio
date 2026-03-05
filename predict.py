import pickle

# Load saved model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# User input
text = input("Enter a sentence: ")

# Transform text
text_vector = vectorizer.transform([text])

# Predict
prob = model.predict_proba(text_vector)[0]
confidence = max(prob)

prediction = model.predict(text_vector)[0]

if prediction == "pos":
    print(f"Sentiment: Positive 😊 (Confidence: {confidence:.2f})")
else:
    print(f"Sentiment: Negative 😡 (Confidence: {confidence:.2f})")
