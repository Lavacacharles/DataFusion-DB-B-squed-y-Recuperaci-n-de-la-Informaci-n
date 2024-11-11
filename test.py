#!/usr/bin/env python3

import pandas as pd
from indice import InvertIndex

# Utilizar la siguiente coleccion de documentos para probar la eficiencia del indice
dataton = pd.read_csv("df_total.csv")
print(dataton.shape)
dataton.head()


def mostrarDocumentos(result):
    for i in result:
        print(i)


index = InvertIndex("indice.dat")
index.building(dataton, 1)  # El texto a procesar esta en la posicion 1

Query1 = "El pais de China y su cooperacion"
result = index.retrieval(Query1, 10)
print("Query1: ", Query1)
mostrarDocumentos(result)

# Proponer 3 consultas adiconales
Query2 = "Los bancos en la actualidad"
print("Query2: ", Query2)
result = index.retrieval(Query2, 10)
mostrarDocumentos(result)

Query3 = "Emergencias en cartagena"
print("Query3: ", Query3)
result = index.retrieval(Query3, 10)
mostrarDocumentos(result)

Query4 = "inteligencia artificial en europa"
print("Query4: ", Query4)
result = index.retrieval(Query4, 10)
mostrarDocumentos(result)

Query5 = "Amenazas para las criptomonedas"
print("Query5: ", Query5)
result = index.retrieval(Query5, 10)
mostrarDocumentos(result)
