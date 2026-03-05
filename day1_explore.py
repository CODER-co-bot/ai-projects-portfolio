import nltk
from nltk.corpus import movie_reviews

# Download dataset (one time)
nltk.download('movie_reviews')

# Basic info
print("Total reviews:", len(movie_reviews.fileids()))

# Example review
file_id = movie_reviews.fileids()[0]
words = movie_reviews.words(file_id)

print("\nSample review (first 40 words):")
print(" ".join(words[:40]))

# Sentiment label
category = movie_reviews.categories(file_id)[0]
print("\nSentiment:", category)
