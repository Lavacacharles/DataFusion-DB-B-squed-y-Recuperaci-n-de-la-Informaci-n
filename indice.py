#!/usr/bin/env python3

import nltk
import numpy as np
from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
import pickle
import re
import os
import math
from collections import defaultdict

# from size import total_size


nltk.download("punkt")

stemmer = SnowballStemmer("spanish")

# In KB, MB or GB
# We use MB
BLOCK_SIZE = 0.5 * 10**7
MEMORY_SIZE = 512 * 10**7
MAX_BLOCKS = MEMORY_SIZE // BLOCK_SIZE
MAX_DOCS = 500

# 1. Definir el stoplist
with open("stoplist.txt", "r", encoding="latin-1") as fil:
    stoplist = [line.strip() for line in fil.readlines()]
stoplist += ["?", "-", ".", ":", ",", "!", ";"]


def preprocesamiento(texto):
    words = []
    # 1- convertir a minusculas
    texto = texto.lower()

    # 2- eliminar signos con regex
    texto = re.sub(r"[^a-zA-Z0-9\sáéíóú]", "", texto)

    # 3- tokenizar
    tokens = nltk.word_tokenize(texto, language="spanish")

    # 3- eliminar stopwords
    tokens = [word for word in tokens if word not in stoplist]

    # 4- Aplicar reduccion de palabras (stemming)
    for i in tokens:
        words.append(stemmer.stem(i))

    return words


class InvertIndex:
    def __init__(self, path):
        self.index_file_count = 0
        self.index_dir = path
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

    def build(self, collection_text, position_text):
        index_number = self.index_file_count
        out_file_path = f"index_{index_number}.dat"
        # the index is the posting list
        # The structure is like:
        # {TERM: [(docID, TF), ...], ...}
        # In place of saving a list in term, we could save a location towards
        # towards a bucket, so that the index only has the terms and a pointer
        # I believe, pointers are neccessary because it's possible that not all
        # entries for a term can fit in only 1 page
        index = defaultdict(list)
        docs = 0

        collections_text_np = collection_text.to_numpy()

        # Iterate
        print("N: ", len(collections_text_np))
        for i in range(len(collections_text_np)):
            print("i: ", i)
            # Add extra to current index
            text_tokenized = preprocesamiento(collections_text_np[i][position_text])
            text_freq = FreqDist(text_tokenized)
            for word, freq in text_freq.items():
                current = (i, np.log(1 + freq))
                # If we pass the limit of the page, we save and create a new index
                # if total_size(index) + total_size(current) > BLOCK_SIZE:
                if docs + 1 > MAX_DOCS:
                    print("index: ", index_number)
                    # we aren't sorting, so it needs to be done when we merge it
                    with open(os.path.join(self.index_dir, out_file_path), "wb") as f:
                        pickle.dump(index, f)
                    index_number += 1
                    out_file_path = f"index_{index_number}.dat"
                    docs = 0
                    index.clear()
                # We still have space, so we add the current value
                index[word].append(current)
                docs += 1

        # If there are remaining elements in the current intdex, we must save
        # them
        if len(index) > 0:
            print("index: ", index_number)
            # we aren't sorting, so it needs to be done when we merge it
            with open(os.path.join(self.index_dir, out_file_path), "wb") as f:
                pickle.dump(index, f)
            index_number += 1
            out_file_path = f"index{index_number}.dat"
            index.clear()
        # Merge
        # TODO
        # In the merge, idf and normalization must be applied at the end, when
        # everythings is in place
        # groups is the number of current partitions. We iterate until groups
        # is 1, meaning all has been merged into 1 group

        groups = index_number // (MAX_BLOCKS - 1)
        start = 0
        count = 1
        # Loops while until the are no more groups
        while groups > 0:
            last_free = 0
            start = 0
            print("groups: ", groups)
            for i in range(groups + 1):
                print("ig: ", i)
                print("start: ", start)
                for j in range(
                    start, min(start + (MAX_BLOCKS - 1) * count, index_number)
                ):
                    print("j: ", j)
                start += (MAX_BLOCKS - 1) * count
            groups //= 2
            count += 1

    def retrieve(self, query, k):
        # TODO
        # Don't take the entire index at once
        self.load_index()
        # lista para el score
        scores = [0] * len(self.length)
        # preprocesar la query: extraer los terminos unicos
        query = preprocesamiento(query)

        # calcular el tf-idf del query
        query_freq = dict(FreqDist(query))
        tfidf_query = {}
        for word, tf in query_freq.items():
            if word in self.idf:
                idf_current = self.idf[word]
                tfidf_query[word] = math.log10(1 + tf) * idf_current

        norm_query = math.sqrt(sum(w_tq**2 for w_tq in tfidf_query.values()))

        # aplicar similitud de coseno
        for term, w_tq in tfidf_query.items():
            if term in self.index:
                for doc, tf_td in self.index[term]:
                    w_td = tf_td * self.idf[term]
                    scores[doc] += w_td * w_tq
        for d in range(len(self.length)):
            if self.length[d] != 0:
                scores[d] /= (
                    self.length[d] * norm_query
                )  # Normalización del documento y la consulta

        # ordenar el score de forma descendente
        result = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        # retornamos los k documentos mas relevantes (de mayor similitud al query)
        return result[:k]

    def load_index(self):
        with open(self.index_file, "rb") as f:
            combined = pickle.load(f)
            self.index = combined["index"]
            self.idf = combined["idf"]
            self.length = combined["length"]
