#!/usr/bin/env python3

import pandas as pd
from indice import InvertIndex

# Utilizar la siguiente coleccion de documentos para probar la eficiencia del indice
dataton = pd.read_csv("lyrics.csv")
print(dataton.shape)


def mostrarDocumentos(result):
    for i in result:
        print(i)


index = InvertIndex("ftest")
print("index created")
index.build(dataton, 1)  # El texto a procesar esta en la posicion 1
print("index built")


# Query1 = "Love is a feeling "
# result = index.retrieval(Query1, 10)
# print("Query1: ", Query1)
# mostrarDocumentos(result)

# Query2 = "Los bancos en la actualidad"
# print("Query2: ", Query2)
# result = index.retrieval(Query2, 10)
# mostrarDocumentos(result)

# Query3 = "Emergencias en cartagena"
# print("Query3: ", Query3)
# result = index.retrieval(Query3, 10)
# mostrarDocumentos(result)

# Query4 = "inteligencia artificial en europa"
# print("Query4: ", Query4)
# result = index.retrieval(Query4, 10)
# mostrarDocumentos(result)
