#!/usr/bin/env python3

from knn_high import load_index, search, create_index, create_index_array, search_time
import pickle
import os


def exists():
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


def create(dataset_path):
    # This creates the index
    index, feature_vectors = create_index_array(dataset_path)

    query_vectors = feature_vectors[:1]
    k = 8

    elapsed_time = search_time(index, query_vectors, k)

    print("Elapsed_time: ", elapsed_time)
    return elapsed_time


folder_path = "reduction/"
archivos_pkl = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]

print(archivos_pkl)

for i in archivos_pkl:
    print("File: ", folder_path + i)
    if os.path.getsize(folder_path + i) > 0:
        create(folder_path + i)
    else:
        print("Empty file")
