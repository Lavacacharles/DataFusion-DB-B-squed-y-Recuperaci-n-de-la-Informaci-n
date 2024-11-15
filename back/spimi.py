import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
import regex as re

stemmer = SnowballStemmer('english')
import os
import pandas as pd

with open("stopwords-en.txt", encoding="latin1") as file:
   stoplist = [line.rstrip().lower() for line in file]
stoplist += ['?', '-', '.', ':', ',', '!', ';']

def preprocesamiento(texto, stemming=True):
  words = []
  texto = str(texto)
  texto = texto.lower()
  texto = re.sub(r'[^a-zA-Z0-9_À-ÿ]', ' ', texto)
  # tokenizar
  words = nltk.word_tokenize(texto, language='spanish')
  # filtrar stopwords
  words = [word for word in words if word not in stoplist]
  # reducir palabras (stemming)
  if stemming:
      words = [stemmer.stem(word) for word in words]
  return words


import os
import pickle
import sys
import math
import heapq

class SPIMI:
    def __init__(self, index_dir="index_blocks", index_dataset="path" , position = 3):
        self.index_dir = index_dir  
        self.path_ = index_dataset  
        self.block_counter = 0      
        self.doc_ids = set()         
        self.idf = {}
        self.length = {}
        self.disk_limit = 4000  
        self.position = position

        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

    def spimi_invert(self, token_stream):
        dictionary = {}

        for doc_id_, row in dataset.iterrows():
            words = preprocesamiento(row.iloc[self.position])
            for text in words:
                doc_id = doc_id_
                token = text
                self.doc_ids.add(doc_id)
                if token not in dictionary:
                    dictionary[token] = {}  

                if doc_id not in dictionary[token]:
                    dictionary[token][doc_id] = 1  
                else:
                    dictionary[token][doc_id] += 1  

                dictionary_size = sys.getsizeof(dictionary)
                if dictionary_size >= self.disk_limit:
                    self.write_block_to_disk(dictionary, level=0)
                    dictionary.clear()
                
            if dictionary:
                self.write_block_to_disk(dictionary, level=0)
        self.load_index()
        self.calculate_idf()
        
    def write_block_to_disk(self, dictionary, level):
        sorted_terms = dict(sorted(dictionary.items())) 
        # block_{it}
        file_path = os.path.join(self.index_dir, f"block_{self.block_counter}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(sorted_terms, f)
        
        self.block_counter += 1

    def load_block(self, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    
    def retrieval(self, query, k):
        self.load_index() 
        N = len(self.doc_ids)  
        scores = [0] * N  
        tf_query = {}  
        terms = preprocesamiento(query)  

        # Calcular el TF-IDF del query
        for term in terms:
            if term in tf_query:
                tf_query[term] += 1
            else:
                tf_query[term] = 1

        tfidf_query = {}
        for term, tf in tf_query.items():
            if term in self.idf:
                tfidf_query[term] = math.log10(1 + tf) * self.idf[term]
        norm_query = math.sqrt(sum(w_tq**2 for w_tq in tfidf_query.values()))  # Normalización del query

        # Aplicar similitud de coseno: Calculamos el puntaje para cada documento
        for term, w_tq in tfidf_query.items():
            if term in self.index:
                for doc, tf_td in self.index[term]:
                    w_td =tf_td* self.idf[term]
                    # print("self.index", term, "es", self.index[term])

                    scores[doc] += w_td * w_tq #Producto punto

        # Normalizar las puntuaciones de los documentos
        for d in range(N):
            if self.length.get(d, 0) != 0:
                scores[d] /= (self.length.get(d, 1) * norm_query)  # Normalización documento y consulta

        # Ordenar las puntuaciones en orden descendente
        result = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        # Devolver los k documentos más relevantes (top-k)
        return result[:k]
    
    def calculate_idf(self):
        for term, postings in self.index.items():
            # Número de documentos que contienen el término
            doc_freq = len(postings)
            # Calcular IDF y almacenar
            self.idf[term] = math.log10(len(self.doc_ids) / (1 + doc_freq))
        # Calcular la longitud de cada documento
        for term, postings in self.index.items():
            for doc_id, tf in postings:
                if doc_id not in self.length:
                    self.length[doc_id] = 0
                # Sumar los TF-IDF al cuadrado para la longitud del documento
                self.length[doc_id] += (tf * self.idf[term]) ** 2
    
        # Tomar la raíz cuadrada para completar la longitud de cada documento
        for doc_id in self.length:
            self.length[doc_id] = math.sqrt(self.length[doc_id])

    def load_index(self):
        self.index = {}
        files = sorted(os.listdir(self.index_dir))
        for file in files:
            if file.endswith(".pkl"):
                block = self.load_block(os.path.join(self.index_dir, file))
                for term, postings in block.items():
                    if term not in self.index:
                        self.index[term] = []
                    for doc_id, tf in postings.items():
                        
                        self.index[term].append((doc_id, math.log10(1+ tf)))
 
    def load_block(self, filepath):
        """ Cargar un bloque del índice desde un archivo. """
        with open(filepath, "rb") as f:
            return pickle.load(f)

def OrdenarPorBloques(dir_blocks, n_blocks):
    for i in range(n_blocks):
        file_path = os.path.join(dir_blocks, 'block_{}.pkl'.format(i))
        with open(file_path, 'rb') as f:
            tuplas_ordenadas = pickle.load(f)
        tuplas_ordenadas = sorted(list(tuplas_ordenadas.items()), key=lambda x: x[0])
        # tuplas_ordenadas = [par for par in tuplas_ordenadas if not par[0].isdigit()]
        tuplas_ordenadas = [(term[0], list(term[1].items())) for term in tuplas_ordenadas]
        with open(file_path, 'wb') as f:
            pickle.dump(tuplas_ordenadas, f)

def mergeSortAux(dir_bloques, l, r):
    if l == r:
        bloque = leer_bloque(dir_bloques, l)
        unicos = set()    
        for par in bloque:
            unicos.add(par[0])
        return list(unicos)
    
    if (l < r):
        mid = int(math.ceil((r + l)/2.0))
        unique_l = mergeSortAux(dir_bloques, l, mid - 1)
        unique_r = mergeSortAux(dir_bloques, mid, r)
        unicos = set()    
        for term in unique_l:
            unicos.add(term)
        for term in unique_r:
            unicos.add(term)
        unicos = list(unicos)
        merge_v2(dir_bloques, l, r, mid, len(unicos))
        return list(unicos)
        
    return []

def escribir_bloque(dir_bloques, block, idx_insert_block, buffer_limit = 2000):
    with open(os.path.join(dir_bloques, "block_{}_v2.pkl".format(idx_insert_block)), 'wb') as f:
        pickle.dump(block, f)    
def leer_bloque(dir_bloques, it):
    file_path = os.path.join(dir_bloques, f"block_{it}.pkl")
    with open(file_path, "rb") as f:
        buffer = pickle.load(f)
    return buffer
def merge_v2(dir_bloques, l, r, mid, num_terms):
    idx_insert_block = l
    new_block = []
    mezclar_n_bloques = r - l + 1
    unique_terms_per_block = int(math.ceil(num_terms/mezclar_n_bloques))
    unique_terms_current_block = 0

    it_l = l
    it_r = mid
    term_dic_l = leer_bloque(dir_bloques, it_l)
    term_dic_r = leer_bloque(dir_bloques, it_r)
    
    idx_term_l = 0
    idx_term_r = 0

    idx_doc_l = 0
    idx_doc_r = 0
    new_block = []
    while(it_l < mid and it_r < r + 1):
        # print(f"Toma 2 bloques {it_l} y {it_r} | idx_term_l: ", idx_term_l, "| len(term_dic_l)", len(term_dic_l), "| idx_term_r: ", idx_term_r, "| len(term_dic_r)", len(term_dic_r))
        while(idx_term_l < len(term_dic_l) and idx_term_r < len(term_dic_r)): # moverme entre palabras de dos bloques
            # print("Current term_dic_l", term_dic_l)
            # print("Current term_dic_r", term_dic_r)
            # print(f"Toma 2 terminos {term_dic_l[idx_term_l][0]} y {term_dic_r[idx_term_r][0]}")
            new_term = []
            if(term_dic_l[idx_term_l][0] < term_dic_r[idx_term_r][0]):
                new_term = term_dic_l[idx_term_l]
                idx_term_l += 1
            elif(term_dic_l[idx_term_l][0] > term_dic_r[idx_term_r][0]):
                new_term = term_dic_r[idx_term_r]
                idx_term_r += 1
            else:
                idx_doc_l = 0
                idx_doc_r = 0
                while(idx_doc_l < len(term_dic_l[idx_term_l][1]) and idx_doc_r < len(term_dic_r[idx_term_r][1])):
                    # print(f"Toma 2 terminos iguales con tf = {term_dic_l[idx_term_l][1]} y {term_dic_r[idx_term_r][1]}")
                    if term_dic_l[idx_term_l][1][idx_doc_l][0] > term_dic_r[idx_term_r][1][idx_doc_r][0]:
                        pushear_doc = term_dic_r[idx_term_r][1][idx_doc_r]
                        idx_doc_r += 1
                    elif term_dic_l[idx_term_l][1][idx_doc_l][0] < term_dic_r[idx_term_r][1][idx_doc_r][0]:
                        pushear_doc = term_dic_l[idx_term_l][1][idx_doc_l]
                        idx_doc_l += 1
                    else:
                        pushear_doc = (term_dic_l[idx_term_l][1][idx_doc_l][0], term_dic_l[idx_term_l][1][idx_doc_l][1] + term_dic_r[idx_term_r][1][idx_doc_r][1])
                        idx_doc_l += 1
                        idx_doc_r += 1
                    new_term.append(pushear_doc)
                while(idx_doc_l < len(term_dic_l[idx_term_l][1])):
                    pushear_doc = term_dic_l[idx_term_l][1][idx_doc_l]
                    idx_doc_l += 1
                    new_term.append(pushear_doc)
                while(idx_doc_r < len(term_dic_r[idx_term_r][1])):
                    pushear_doc = term_dic_r[idx_term_r][1][idx_doc_r]
                    idx_doc_r += 1
                    new_term.append(pushear_doc)
                new_term = (term_dic_l[idx_term_l][0], new_term)
                idx_term_r += 1
                idx_term_l += 1
            new_block.append(new_term)
            
            unique_terms_current_block += 1
            if (unique_terms_current_block == unique_terms_per_block):
                escribir_bloque(dir_bloques, new_block, idx_insert_block)
                unique_terms_current_block = 0
                idx_insert_block += 1
                new_block = []
        if(len(term_dic_l) == idx_term_l):
            if (it_l < mid - 1):
                it_l += 1
                term_dic_l = leer_bloque(dir_bloques, it_l)
                idx_term_l = 0
                idx_doc_l = 0
                continue
            else:
                break
        if(len(term_dic_r) == idx_term_r):
            if (it_r < r):
                it_r += 1
                term_dic_r = leer_bloque(dir_bloques, it_r)
                idx_term_r = 0
                idx_doc_r = 0
                continue
            else:
                break
        if(it_l == mid | it_r == r + 1):
            break
    while(it_l < mid):
        term_dic_l = leer_bloque(dir_bloques, it_l)
        while(idx_term_l < len(term_dic_l)):
            new_block.append(term_dic_l[idx_term_l])
            unique_terms_current_block += 1
            if (unique_terms_current_block == unique_terms_per_block):
                escribir_bloque(dir_bloques, new_block, idx_insert_block)
                unique_terms_current_block = 0
                idx_insert_block += 1
                new_block = []
            idx_term_l += 1
        idx_term_l = 0
        it_l += 1
    while(it_r < r + 1):
        term_dic_r = leer_bloque(dir_bloques, it_r)
        while(idx_term_r < len(term_dic_r)):
            new_block.append(term_dic_r[idx_term_r])
            unique_terms_current_block += 1
            if (unique_terms_current_block == unique_terms_per_block):
                escribir_bloque(dir_bloques, new_block, idx_insert_block)
                unique_terms_current_block = 0
                idx_insert_block += 1
                new_block = []
            idx_term_r += 1
        idx_term_r = 0
        it_r += 1

    while(idx_insert_block < r + 1):
        if len(new_block) > 0:
            escribir_bloque(dir_bloques, new_block, idx_insert_block)
        else:
            escribir_bloque(dir_bloques, [], idx_insert_block)
        new_block = []
        idx_insert_block += 1
    idx_insert_block = l
    for idx_archivo in range(l, r + 1):
        nuevo_nombre = os.path.join(dir_bloques, "block_{}.pkl".format(idx_archivo))
        if os.path.exists(nuevo_nombre):
            os.remove(nuevo_nombre)
        os.rename(os.path.join(dir_bloques, "block_{}_v2.pkl".format(idx_archivo)), nuevo_nombre)
def mergeSort(dir_bloques):
    bloques_files_dir = os.listdir(os.path.join('./',dir_bloques))
    # print(bloques_files_dir)
    n = len(bloques_files_dir)
    # n = int(math.exp2(math.floor(math.log2(n)) + 1))
    mergeSortAux(dir_bloques, 0, n - 1)
'''
Recibe la direccion del folder con los diccionarios del spimi,
modifica los archivos y los sobreescribe para crear el índice
invertido con un índice global
'''
def InvertirListasDiccionarios(dir_bloques):
    n_bloques = len(os.listdir(os.path.join('./',dir_bloques)))
    for i in range(n_bloques):
        bloque_path = os.path.join(dir_bloques,"block_{}.pkl".format(i))
        with open(bloque_path, 'rb') as f:
            bloque = pickle.load(f)
        bloque_dict = {}
        if len(bloque) != 0:
            for term, poosting_list in bloque:
                poosting_dict = {}
                for doc, tf in poosting_list:
                    poosting_dict[doc] = tf
                bloque_dict[term] = poosting_dict
        with open(bloque_path, 'wb') as f:
            pickle.dump(bloque_dict, f)

def getNumberWithAtributo(dataset_head, atributo):
    if atributo in dataset_head.columns:
        return dataset_head.columns.get_loc(atributo)  # Obtiene la posición del atributo
    return -1  # Retorna -1 si no existe
 

# START OF THE MAIN CODE

 # --------Set Path----------
import kagglehub
path = kagglehub.dataset_download("imuhammad/audio-features-and-lyrics-of-spotify-songs")

# ----------Set dataset--------

lista_ = os.listdir(path)
songs = os.path.join(path, lista_[0])
print("Path de songs: ",songs)
dataset = pd.read_csv(songs)
dataset = dataset.head(20)

# ----------Set Atributo--------
columna = getNumberWithAtributo(dataset, "lyrics")

# ---------Creation---------
s = SPIMI("index_blocks", path, columna)  
s.spimi_invert(dataset)
OrdenarPorBloques(s.index_dir, s.block_counter )
mergeSort(s.index_dir)
InvertirListasDiccionarios("index_blocks")

# ---------End Creation---------

def getResultados(result, path):
    res = []
    lista_ = os.listdir(path)
    dataset_path_ = os.path.join(path, lista_[0])
    dataset = pd.read_csv(dataset_path_)
    for doc, score in result:
        res.append((dataset.iloc[doc], score))
    return pd.DataFrame(res)    

query = dataset.iloc[19, 3]
top_k = 5

#-----------Start of query------------
result = s.retrieval(query, top_k)
getResultados(result, s.path_) # La respuesta es un array pd de [fila del dataframe, score]
#-----------End of query------------

