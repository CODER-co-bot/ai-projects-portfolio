import pickle
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
from sklearn.metrics import accuracy_score, classification_report

# Download dataset (safe if already downloaded)
nltk.download('movie_reviews')

# ----------------------------
# 1. Load data
# ----------------------------
documents = []
labels = []

for fileid in movie_reviews.fileids():
    words = movie_reviews.words(fileid)
    text = " ".join(words)
    documents.append(text)

    label = movie_reviews.categories(fileid)[0]
    labels.append(label)

print("Total samples:", len(documents))

# ----------------------------
# 2. Convert text → numbers (TF-IDF)
# ----------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=10000,
    ngram_range=(1,2)
)


X = vectorizer.fit_transform(documents)
y = labels

# ----------------------------
# 3. Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4. Train model
# ----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ----------------------------
# 5. Evaluate
# ----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# 6. Save model and vectorizer
# ----------------------------
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")

