#!/usr/bin/env python3

import re


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
    # To tokenize and the rest of the preprocesing  is done when executed by to_tsquery,
    # so doing it here would be double work

    return texto


# Example
preprocesamiento("you are like me")
