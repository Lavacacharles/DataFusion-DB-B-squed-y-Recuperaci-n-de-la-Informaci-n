import nltk
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import nltk


nltk.download('punkt')
nltk.download('punkt_tab')


import regex as re
import os
import pandas as pd
import os
import pickle
import sys
import math
import dbm
import time
import heapq
import json

stemmerEnglish = SnowballStemmer('english')
stemmerSpanish = SnowballStemmer('spanish')

def preprocesamiento(language, texto, stemming=True):

    filenameStopword = "stopwords-en.txt"
    stemmer = stemmerEnglish

    if language == 'spanish':
        filenameStopword = "stoplist.txt"
        stemmer = stemmerSpanish
    
        
    with open(filenameStopword, encoding="latin1") as file:
        stoplist = [line.rstrip().lower() for line in file]
    stoplist += ['?', '-', '.', ':', ',', '!', ';']


    words = []
    texto = str(texto)
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z0-9_À-ÿ]', ' ', texto)

    words = nltk.word_tokenize(texto, language)

    words = [word for word in words if word not in stoplist]

    if stemming:
        words = [stemmer.stem(word) for word in words]
    return words


# Funciones AUXILIARES para recuperar la información

def getNumberWithAtributo(dataset_head, atributo):
    if atributo in dataset_head.columns:
        return dataset_head.columns.get_loc(atributo)  # Obtiene la posición del atributo
    return -1  # Retorna -1 si no existe
 

def getResultados(result, path):
    df = pd.read_csv(path)  # Reemplaza 'documento.csv' con el nombre de tu archivo
    # Crear un DataFrame a partir del resultado
    result_df = pd.DataFrame(result, columns=['id', 'score'])
    # Asumir que 'id' corresponde al índice del DataFrame original
    # Si 'id' es una columna específica en tu CSV, ajusta 'left_on' y 'right_on' en el merge
    df = df.reset_index().rename(columns={'index': 'id'})

    # Unir ambos DataFrames basándose en 'id'
    merged_df = pd.merge(result_df, df, on='id', how='left')

    # Reemplazar valores NaN en 'score' con 0 (si es necesario)
    merged_df['score'] = merged_df['score'].fillna(0)

    # Opcional: Reordenar las columnas si lo deseas
    # merged_df = merged_df[df.columns.tolist() + ['score']]

    # Convertir el DataFrame resultante a una cadena CSV con separador punto y coma
    csv_string = merged_df.to_csv(sep='%', index=False)
 
    return csv_string

def getResultadosDF(result, path, disk_limit):
    res = []
    for chunk in pd.read_csv(path, chunksize=disk_limit):
        for doc, score in result:
            row = chunk.iloc[doc].copy()
            row['score'] = score  
            res.append(row)
    return res 
def get_dfTex_Cols(path, row, columnas): #We don´t use it is only ofr simple an very fast testing
    dataset =  pd.read_csv(path)
    res = [dataset.iloc[row, i] for i in columnas]
    return ' '.join(res)  
#Funciones auxiliares para el mergeo:

class SPIMI:
    def __init__(self, index_dataset_path="path",bloques_dir="index_blocks" , columnas = ["track_name","track_artist","lyrics", "track_album_name"], language='english'):
        self.bloques_dir = bloques_dir  
        self.dataset_path = index_dataset_path  
        self.block_counter = 0      
        self.doc_ids = None  
        self.idf = {}
        self.length = {}
        self.disk_limit = 40000
        self.language = language  
        
        #--Columnas:
        self.columnas = columnas 
        # ----Direciones de memoria para los files--
        self.dirIndex = self.bloques_dir + "_merge"
        
        self.term_dict_path = os.path.join(self.dirIndex, 'term_dict.pkl')
        self.postings_file_path = os.path.join(self.dirIndex, 'postings.bin')
        self.norms_file_path = os.path.join(self.dirIndex, 'document_norms.pkl')
        #Guardar los ids
        self.doc_count_path = os.path.join(self.dirIndex, 'doc_count.txt')  

        if not os.path.exists(self.dirIndex):
            os.makedirs(self.dirIndex)
        if not os.path.exists(self.bloques_dir):
            os.makedirs(self.bloques_dir)

        if os.path.exists(self.doc_count_path): # Guardamos el total de docs en ves de usar un set
            with open(self.doc_count_path, 'r') as f:
                self.doc_ids = int(f.read().strip())

        self.load_doc_ids()   

    def load_doc_ids(self):
        if os.path.exists(self.doc_count_path):
            try:
                with open(self.doc_count_path, 'r') as f:
                    self.doc_ids = int(f.read().strip())
                    print(f"Documentos cargados: {self.doc_ids}")
            except (ValueError, FileNotFoundError):
                self.doc_ids = 0  # Si el archivo está vacío o corrupto
        else:
            self.doc_ids = 0
 
       
        
                  

    def spimi_invert(self):
        print("Creando y ordenando cada bloque")
        dictionary = {}
        doc_ids = set()
        time_spimiInvert_start = time.time()

        #Get columna indeces: 
        with open(self.dataset_path, mode='r', encoding='utf-8') as file:
            primera_linea = file.readline().strip()
        #Get number columns
        columnas = primera_linea.split(',')
        columnas_numbers  = [i for i in range(len(columnas)) if columnas[i] in self.columnas]

        # Creamos los diccionarios a los bloques
        for chunk in pd.read_csv(self.dataset_path,chunksize= self.disk_limit): 
            for doc_id_, row in chunk.iterrows():
                self.doc_ids +=1 # Actualizamos la cantidad de id
                preFila = [str(row.iloc[i]) for i in columnas_numbers]
                texto = ' '.join(item for item in preFila)
                words = preprocesamiento(self.language,texto)
                for text in words:
                    doc_id = doc_id_
                    token = text
                    if token not in dictionary:
                        dictionary[token] = {}  

                    if doc_id not in dictionary[token]:
                        dictionary[token][doc_id] = 1  
                    else:
                        dictionary[token][doc_id] += 1  

                    dictionary_size = sys.getsizeof(dictionary)
                    if dictionary_size >= self.disk_limit:
                        self.write_block_to_disk(dictionary)
                        dictionary.clear()

        if dictionary:
            self.write_block_to_disk(dictionary)  
        
        with open(self.doc_count_path, 'w') as f:
            f.write(str(self.doc_ids))

        time_spimiInvert = time.time()- time_spimiInvert_start

        # MERGE
        print("Iniciando con el merge")
        time_Merge_start = time.time()
        self.mergeHeap()
        time_Merge = time.time() - time_Merge_start


        print("\nResumen de tiempos:")
        print(f"Tiempo para la creación Creación  y el ordenamiento de los bloques: {time_spimiInvert:.2f} segundos")
        print(f"Tiempo para la ejecución del merge: {time_Merge:.2f} segundos")
        print(f"Tiempo total para la creación y el mergeo: {time_Merge + time_spimiInvert:.2f} segundos")
    


    def write_block_to_disk(self, dictionary): # Guardo los ordenados
        # TO DO
        # Probar con guardar y luego ordenar con hilos
        sorted_terms = dict(sorted(dictionary.items())) 
        file_path = os.path.join(self.bloques_dir, f"block_{self.block_counter}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for term, postings in sorted_terms.items():
                postings_str = json.dumps(postings)  
                f.write(f"{term}: {postings_str}\n")  
        self.block_counter += 1


    def load_term_dict(self):
        with open(self.term_dict_path, 'rb') as f:
            while True:
                try: #Yiel es un generador, en vez de un return yield mantienes q en piclek esta 
                    yield pickle.load(f) #Avanzamos bloque por bloque, es decir dicionario por diccionario
                except EOFError:
                    break


    def show_Block(self, index):
        print("Imprimiendo contenido del bloque: ", index)
        file_path = os.path.join(self.bloques_dir, f'block_{index}.pkl')
        if not os.path.exists(file_path):
            print(f"El bloque {index} no existe en la ruta: {file_path}")
            return
        try:
            with open(file_path, 'rb') as f:
                block_content = pickle.load(f)
            print(block_content)
        except Exception as e:
            print(f"Error al abrir el bloque {index}: {e}")



    # Ya se comprobó el correcto merge 
    # Ahora los archivo son terminos postings y salto de linea, asi se getea linea por línea 
    def mergeHeap(self):
        lista_bloques = os.listdir(self.bloques_dir)
        priority_queue = []
        idx = 0 
    
        postings_file_path = os.path.join(self.dirIndex, 'postings.bin')
        term_dict_path = os.path.join(self.dirIndex, 'term_dict.pkl')
        norms_file_path = os.path.join(self.dirIndex, 'document_norms.pkl')
    
        open_files = {}
        doc_norms_temp = {}  
    
        #Para optimizar en término s de manejo de memoria secundaria a term_dict.pkl haremos lo siguiente: 
        # en un diccionario temporal y si se supera a disk_limit se guarda
        # las el mañao del diccionario y le diccionario, de tal forma que luego se puda abrir por bloques
        # En términos prácticos es (pos, tamanio) diccionario tempora ...
        temp_term_dict = {}
        temp_term_dict_position = 0

        try:
            for bloque_path in lista_bloques:  # Referencias de bloques y pusheo de una línea (term, postings)
                bloque_full_path = os.path.join(self.bloques_dir, bloque_path)
                open_files[bloque_path] = open(bloque_full_path, "r", encoding="utf-8")
                term_postings = self.read_next_term(open_files[bloque_path])
                if term_postings:
                    term, postings = term_postings
                    heapq.heappush(priority_queue, (term, idx, postings, bloque_path))
                    idx += 1
    
            current_term = None
            current_postings = {}
            postings_file_position = 0
    
            with open(postings_file_path, 'wb') as postings_file, \
                 open(term_dict_path, 'ab') as term_dict_file:
    
                while priority_queue:
                    term, _, postings, bloque_path = heapq.heappop(priority_queue)
                    file = open_files[bloque_path]
    
                    if term == current_term:  # Caso en que el término es igual
                        for doc_id, tf in postings.items():
                            current_postings[doc_id] = current_postings.get(doc_id, 0) + tf
                    else:  # Nuevo término
                        if current_term is not None:
                            # Calcular IDF
                            idf = math.log10(self.doc_ids / (1 + len(current_postings)))
    
                            # Guardar postings (TF e IDF) en postings.bin
                            tf_idf_postings = {}
                            for doc_id, tf in current_postings.items():
                                tf_weighted = math.log10(1 + tf)
                                tfidf = tf_weighted * idf
                                tf_idf_postings[doc_id] = {"tf": tf_weighted, "idf": idf}
    
                                # Acumular el cuadrado del TF-IDF en las normas temporales
                                doc_norms_temp[doc_id] = doc_norms_temp.get(doc_id, 0) + tfidf ** 2
    
                            postings_data = pickle.dumps(tf_idf_postings)
                            postings_file.write(postings_data)
    
                            # Guardar término en term_dict_temporal: 
                            temp_term_dict[current_term] =  (postings_file_position, len(postings_data)) 
                            postings_file_position += len(postings_data)
                        
                        # Si se puera el limite ------><------
                        dictionary_size = sys.getsizeof(temp_term_dict)
                        if dictionary_size >= self.disk_limit: 
                            pickle.dump(temp_term_dict, term_dict_file) 
                            
                            temp_term_dict.clear()
                        
                        # Actualizar current_term y current_postings
                        current_term = term
                        current_postings = postings.copy()

                    next_term_postings = self.read_next_term(file)
                    if next_term_postings:
                        next_term, next_postings = next_term_postings
                        heapq.heappush(priority_queue, (next_term, idx, next_postings, bloque_path))
                        idx += 1
    
                
                if current_term is not None:
                    idf = math.log10(self.doc_ids / (1 + len(current_postings)))
                    tf_idf_postings = {}
                    for doc_id, tf in current_postings.items():
                        tf_weighted = math.log10(1 + tf)
                        tfidf = tf_weighted * idf
                        tf_idf_postings[doc_id] = {"tf": tf_weighted, "idf": idf}
    
                        # Acumular el cuadrado del TF-IDF en las normas temporales
                        doc_norms_temp[doc_id] = doc_norms_temp.get(doc_id, 0) + tfidf ** 2

                    

                    postings_data = pickle.dumps(tf_idf_postings)
                    postings_file.write(postings_data)
                    temp_term_dict[current_term] = (postings_file_position, len(postings_data))
                    # pickle.dump({current_term: (postings_file_position, len(postings_data))}, term_dict_file)
                    postings_file_position += len(postings_data)

                    #Guardamos de frente temp_term_dict
                    pickle.dump(temp_term_dict, term_dict_file) #Guardo el diccionario
                    temp_term_dict.clear()

                    sorted_terms = list(temp_term_dict.keys())
                    
                    temp_term_dict.clear()
    
        finally:
            for file in open_files.values():
                file.close()
    
        # Guardar las normas finales tomando la raíz cuadrada
        final_norms = {int(doc_id): math.sqrt(value) for doc_id, value in doc_norms_temp.items()}
        with open(norms_file_path, 'wb') as norms_file:
            pickle.dump(final_norms, norms_file)
    
        self.term_dict_path = term_dict_path
        self.postings_file_path = postings_file_path
        print(f"Índice fusionado guardado en {postings_file_path}")
        print(f"Diccionario de términos guardado en {term_dict_path}")
        print(f"Normas de documentos guardadas en {norms_file_path}")
            # Realiza merge de todos los bloques:
            # Manejaremos un archivo para ubicar los un indice general self.indexTerms donde se guarda la posición del termino
            # En el otro archivo se guardaran completito tanto la palabra como el posting list pero ya ordenado
    
    def read_next_term(self, file):
        line = file.readline()
        if not line:  # Si llegamos al final del archivo
            return None
        term, postings_str = line.strip().split(":", 1)
        postings = json.loads(postings_str)  
        return term, postings

    #--Funciones para testing--
    def print_all_norms(self):
        if not os.path.exists(self.norms_file_path):
            print("El archivo de normas no existe. Asegúrate de haber ejecutado 'spimi_invert' correctamente.")
            return

        try:
            with open(self.norms_file_path, 'rb') as norms_file:
                norms = pickle.load(norms_file)
                if not norms:
                    print("El archivo de normas está vacío.")
                    return

                print("Normas de los documentos:")
                for doc_id, norm in sorted(norms.items()):
                    print(f"Documento ID: {doc_id}, Norma: {norm}")
        except Exception as e:
            print(f"Error al leer o procesar las normas: {e}")


    def show_terms_and_positions(self):
        print("Términos y sus posiciones en el archivo de postings:")
        all_terms = []  
        # for term, (position, length) in term_dict.items():
        #     print(f"Término: '{term}', Posición: {position}, Longitud: {length}")

        for term_dict in self.load_term_dict():  
            terms = list(term_dict.keys())
            all_terms.extend(terms)
        if all_terms == sorted(all_terms):
            print("----->Yeeeei, Los términos están ordenados alfabéticamente en el diccionario final.")
        else:
            print("Los términos NO están ordenados alfabéticamente en el diccionario final.")


    def show_terms_with_postings(self):
        if not os.path.exists(self.postings_file_path):
            print("El archivo de postings no existe. Ejecuta 'mergeHeap' primero.")
            return

        print("Términos y sus listas de postings:")
        try:
            with open(self.postings_file_path, 'rb') as postings_file:
                for term_dict in self.load_term_dict(): 
                    for term, (position, length) in term_dict.items():
                        postings_file.seek(position)  
                        postings_data = postings_file.read(length) 
                        postings = pickle.loads(postings_data) 
                        print(f"Término: '{term}', Postings: {postings}")
        except Exception as e:
            print(f"Error al mostrar los términos y postings: {e}")


    def retrieval(self, query, k):
        # Cargamos doc_ids si no está ya cargado
        if self.doc_ids is None:
            self.load_doc_ids()
    
        if self.doc_ids == 0:  
            raise ValueError("El índice no está construido o no contiene documentos. Ejecuta 'spimi_invert' antes de realizar consultas.")
        print("Se cargó en total: ", self.doc_ids, " documentos")
    
        # Preprocesar la consulta
        terms = preprocesamiento(self.language,query)
    
        tf_query = {}
        for term in terms:
            if term in tf_query:
                tf_query[term] += 1
            else:
                tf_query[term] = 1
    
        tfidf_query = {}
        norm_query = 0
    
        scores = [0] * self.doc_ids  
    
        with open(self.postings_file_path, 'rb') as postings_file, \
             open(self.norms_file_path, 'rb') as norms_file:
    
            for term_dict_block in self.load_term_dict(): # bloque por bloque gracias a yield
                for term, tf in tf_query.items():
                    if term in term_dict_block:  # Considerar términos presentes en este bloque
                        position, length = term_dict_block[term]
    
                        postings_file.seek(position)
                        postings_data = postings_file.read(length)
                        postings = pickle.loads(postings_data)  # Diccionario {doc_id: {'tf': tf, 'idf': idf}}
                        idf = math.log10(self.doc_ids / (1 + len(postings)))
    
                        tfidf_query[term] = math.log10(1 + tf) * idf
                        norm_query += (tfidf_query[term]) ** 2
    
                        w_tq = tfidf_query[term]
                        for doc_id_str, values in postings.items():
                            doc_id = int(doc_id_str)
                            tfidf_td = values['tf'] * values['idf']
                            scores[doc_id] += tfidf_td * w_tq 
    
            norm_query = math.sqrt(norm_query)
    
            norms_file.seek(0)
            norms = pickle.load(norms_file)
    
            for doc_id in range(self.doc_ids):
                doc_norm = norms.get(doc_id, 0)
                if doc_norm != 0:
                    scores[doc_id] /= (doc_norm * norm_query)
    
        result = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return result[:k]
# Path

# Create table tablita from path "takataka"
# ---------Creation---------
#path = "spotify_2000.csv"
#--Columnas de indexacion: 
# Select "track_name","track_artist","lyrics", "track_album_name"
#columnas = ["lyrics"]

#columnas = ['lyrics', 'track_name']
#s = SPIMI(path, columnas=columnas, language='spanish')  
#s.spimi_invert() 

# ---------End Creation---------
#query = "El amor es una magia"
#top_k = 5

#result = s.retrieval(query, top_k)
#print(getResultados(result, path))
