#!/usr/bin/env python3

import nltk
from nltk.stem.snowball import SnowballStemmer
import re

nltk.download("punkt")


stemmer = SnowballStemmer("spanish")

# 1. Definir el stoplist
with open("stoplist.txt", "r", encoding="latin-1") as fil:
    stoplist = [line.strip() for line in fil.readlines()]
stoplist += ["?", "-", ".", ":", ",", "!", ";"]


def preprocesamiento(texto):
    # 1- convertir a minusculas
    texto = texto.lower()

    # 2- eliminar signos con regex
    texto = re.sub(r"[^a-zA-Z0-9\sáéíóú]", "", texto)

    # Replace spaces with "|"", the or operator to send it to ts_query
    texto = texto.replace(" ", "|")

    # The ' symbol is neccessary when passing a string to postgresql
    texto = "'" + texto + "'"

    # 3- tokenizar
    # To tokenize and what happens below is done when executed by to_tsquery,
    # so doing it here would be double work
    # tokens = nltk.word_tokenize(texto, language="spanish")

    # 3- eliminar stopwords
    # tokens = [word for word in tokens if word not in stoplist]

    # 4- Aplicar reduccion de palabras (stemming)
    # for i in tokens:
    # words.append(stemmer.stem(i))

    return texto


# Example
# preprocesamiento(
#     "I'm the greatest rapper alive. So damm great, I died. And they thought I was dead"
# )
