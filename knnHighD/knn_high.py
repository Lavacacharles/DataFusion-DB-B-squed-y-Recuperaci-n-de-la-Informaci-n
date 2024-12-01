#!/usr/bin/env python3


import pickle
import numpy as np
import faiss


def create_index(dataset_path, gpu=False, save=False):
    """
    The dataset path leads to a pkl file of the form
    [{"image_name": NAME, "embedding": FEATURE_VECTOR}, ...]
    If a GPU is available, send True
    """
    # 1. Opens the file and extracts the data
    with open(dataset_path, "rb") as file:
        dataset_nested = pickle.load(file)

    # 2. Puts the information of the file into numpy arrays
    image_names = []
    feature_vectors = []

    for i in dataset_nested:
        if len(i["embedding"]) == 0:
            continue
        image_names.append(i["image_name"])
        feature_vectors.append(i["embedding"][0])

    feature_vectors = np.array(feature_vectors)
    feature_vectors = feature_vectors.astype("float32")

    d = feature_vectors.shape[1]

    # Normalize to use cosine similarity
    faiss.normalize_L2(feature_vectors)

    ## Using an IVF index
    nlist = 100
    quantizer = faiss.IndexFlatIP(d)  # the other index
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    # The inner-product search is for the cosine similarity

    if gpu:
        res = faiss.StandardGpuResources()  # use a single GPU
        # make it an IVF GPU index
        gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)
        if not gpu_index_ivf.is_trained:
            # train with the vectors. Needs to be done because the gpu is used
            gpu_index_ivf.train(feature_vectors)

        gpu_index_ivf.add(feature_vectors)  # add vectors to the index

        index_ivf = gpu_index_ivf
    else:
        index_ivf.add(feature_vectors)

    if save:
        faiss.write_index(index_ivf, "index.faiss")
        with open("image_names.pkl", "wb") as file:
            pickle.dump(image_names, file)

    return index_ivf, image_names


def load_index(index_path, image_names_path, gpu=False):
    """
    Loads the index and the image names from files
    If a gpu is available, it can be loaded there
    """
    index = faiss.read_index(index_path)
    if gpu:
        res = faiss.StandardGpuResources()  # use a single GPU
        index = faiss.index_cpu_to_gpu(res, 0, index)

    with open(image_names_path, "rb") as file:
        image_names = pickle.load(file)
    return index, image_names


def search(index, image_names, query, k):
    """
    The query represents a list of feature vectors. It's a list, because faiss
    accepts several queries to search at once
    The index must be a faiss index
    image_names is a list of the image names
    """
    # Preprocess and normalize the query vector
    query_vector = np.array(query)
    query_vector = query_vector.astype("float32")
    faiss.normalize_L2(query_vector)

    # The first found is always itself, so we add 1 more to get k elements
    D, I = index.search(query_vector, k + 1)
    paths = []
    for query_idx, neighbors in enumerate(I):
        current = []
        for rank, neighbor_idx in enumerate(neighbors):
            current.append(image_names[neighbor_idx])
    paths.append(current[1:])

    return paths


def search_image(index, image_names, image, k):
    # Turn image into feature vector
    # TODO
    image_feature_vectors = []

    # apply_normal search
    # The image_feature_vector is put inside a list, because that is how it works
    paths = search(index, image_names, image_feature_vectors, k)
    return paths
