#!/usr/bin/env python3

from knn_high import load_index, search
import pickle
import numpy as np
import time


index, image_names = load_index("index.faiss", "image_names.pkl")

dataset_path = "the_last_one.pkl"
# Open a file and use dump()
with open(dataset_path, "rb") as file:
    # A new file will be created
    dataset_nested = pickle.load(file)

feature_vectors = []

for i in dataset_nested:
    if len(i["embedding"]) == 0:
        continue
    feature_vectors.append(i["embedding"][0])

feature_vectors = np.array(feature_vectors)
feature_vectors = feature_vectors.astype("float32")

query_vectors = feature_vectors[:1]
k = 8

start = time.time()
paths = search(index, image_names, query_vectors, k)
end = time.time()

print("Paths: ", paths)
print("Elapsed_time: ", end - start)
