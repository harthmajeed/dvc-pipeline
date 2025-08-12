from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sys

docs = [
    "I love cats",
    "Cats love dogs",
    "Birds fly high above trees",
    "The dog chased the ball"
] * 1000  # Repeat to make bigger dataset

vectorizer = CountVectorizer()
X_sparse = vectorizer.fit_transform(docs)

# Memory usage of sparse matrix
sparse_mem = (X_sparse.data.nbytes + 
              X_sparse.indptr.nbytes + 
              X_sparse.indices.nbytes)

# Convert to dense matrix
X_dense = X_sparse.toarray()
dense_mem = X_dense.nbytes

print(f"Sparse shape: {X_sparse.shape}")
print(f"Sparse size: {sparse_mem/1024:.2f} KB")
print(f"Dense size: {dense_mem/1024:.2f} KB")

