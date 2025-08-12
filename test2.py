from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

docs = [
    "I love cats",
    "Cats love dogs",
    "Birds fly high above trees",
    "The dog chased the ball"
] * 1000

vec = CountVectorizer(
    token_pattern=r"(?u)\b\w+\b",     # keep single-letter tokens like "I"
    stop_words="english",             # drop common words
    ngram_range=(1, 2),               # unigrams + bigrams
    max_features=20000
)

X = vec.fit_transform(docs)

sparse_mem = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
dense_mem = X.toarray().nbytes

print("Shape:", X.shape)
print("Sparse size KB:", round(sparse_mem/1024, 2))
print("Dense size KB:", round(dense_mem/1024, 2))

