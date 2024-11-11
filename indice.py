#!/usr/bin/env python3

import nltk
import numpy as np
from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
import pickle
import re
import math
from collections import defaultdict


nltk.download("punkt")

stemmer = SnowballStemmer("spanish")

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
    def __init__(self, index_file):
        self.index_file = index_file
        self.index = {}
        self.idf = {}
        self.length = {}

    def building(self, collection_text, position_text):
        # TODO
        # Use SPIMI
        # A different version of the algorithm can be used with secondary memory

        collections_text_np = collection_text.to_numpy()
        # build the inverted index with the collection

        # compute the tf
        tf = []
        self.index = defaultdict(list)
        for i in range(len(collections_text_np)):
            texto = collections_text_np[i][position_text]
            texto = preprocesamiento(texto)

            # calculate tf
            text_freq = FreqDist(texto)
            doc_tf = []
            for word, freq in text_freq.items():
                self.index[word].append((i, np.log(1 + freq)))
                doc_tf.append(np.log(1 + freq))
            tf.append(doc_tf)
        # print(self.index["empresarial"])

        # compute the idf
        for key, value in self.index.items():
            self.idf[key] = np.log(len(collections_text_np) / len(value))
        # print(self.idf)

        # compute the length (norm)
        for doc_id in range(len(collection_text)):
            norm = 0
            for term, postings in self.index.items():
                for doc, tf in postings:
                    if doc == doc_id:
                        norm += (tf * self.idf[term]) ** 2
            self.length[doc_id] = math.sqrt(norm)
        # store in disk
        combined = {"index": self.index, "idf": self.idf, "length": self.length}
        with open(self.index_file, "wb") as f:
            pickle.dump(combined, f)

    def retrieval(self, query, k):
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
