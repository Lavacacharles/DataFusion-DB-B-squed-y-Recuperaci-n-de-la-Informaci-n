from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from parser import process_query
import psycopg2
from psycopg2 import sql
from psycopg2.extras import NamedTupleCursor
from spimi import SPIMI, getResultados
from parser import ManageExceptions
from knnSeq import knnSequentialSearch, ImageSearcher
import hashlib
import os
import csv
import io
import json
import re

def dbconnection(database):
    conection = psycopg2.connect(**database)
    return conection

def preprocesamientoQuery(texto):
    
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Z0-9\sáéíóú]", "", texto)
    texto = texto.replace(" ", "|")
    texto = "'" + texto + "'"
    return texto

def generate_update_query(schema,language, table_name, vector_name, columns):

    weights = ['A', 'B', 'C', 'D']
    setweight_clauses = []
    
    for i, column in enumerate(columns):
        weight = weights[i % len(weights)]
        setweight_clause = (
            f"setweight(to_tsvector('{language}', COALESCE({column}, '')), '{weight}')"
        )
        setweight_clauses.append(setweight_clause)
    
    setweight_string = " ||\n            ".join(setweight_clauses)   
    query = f"""
        UPDATE {schema}.{table_name} SET {vector_name} =
            {setweight_string};
    """
    return query

class CompilerExecute:
    def __init__(self):
        self.last_error = None
        self.data_config = {
            'host': '127.0.0.1',
            'database': 'workiretrieval',
            'user': 'admin_if',
            'password': 'administrator@if$',
            'port': 5432
        }

    def create_table(self, schema, query, file):
        columns = ""
        columnsStatement = ""
        for column in query['columns']:
            if columns != "":
                columns += ","
                columnsStatement += ","
            columnsStatement += f"{column['name']} {column['type']}"
            columns += f"{column['name']}"
        sqlStatement = f'''
            CREATE TABLE {schema}.{query['table_name']}(
                {columnsStatement}
            );
        '''
        print(f"Creating table: {query['table_name']}")
        print("Columns:")
        print(f"Source file: {query['source_file']}")

        try:
            connection = dbconnection(self.data_config)
            cursor = connection.cursor()
            cursor.execute(sqlStatement)
            connection.commit()
        except Exception as e:
            print(f"Error al crear la tabla: {e}")
            connection.rollback()
            cursor.close()
            connection.close()
            raise ManageExceptions(e)

        try:
            with open(file, 'r', encoding='utf-8') as f:
                sql_copy = sql.SQL(f"""
                    COPY {schema}.{query['table_name']}({columns})
                    FROM STDIN WITH (FORMAT csv, HEADER true)
                """)
                cursor.copy_expert(sql_copy, f)
                connection.commit()
        except Exception as e:
            print(f"Error al copiar los datos: {e}")
            connection.rollback()
            cursor.close()
            connection.close()
            raise ManageExceptions("Ups! Tabla creada correctamente, pero error en cargar los datos")

        try:
            cursor.execute(f'''
                SELECT track_name, track_artist, track_album_name, playlist_name, language
                FROM {schema}.{query['table_name']}
                ORDER BY track_id
                LIMIT 50;
            ''')
            rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]

            output = io.StringIO()
            writer = csv.writer(output, delimiter='%')
            writer.writerow(column_names)  
            writer.writerows(rows)         
            csv_text = output.getvalue()
            output.close()
            print("Primeros 50 registros en formato CSV:")
            return csv_text

        except Exception as e:
            print(f"Error al recuperar los registros: {e}")
            raise ManageExceptions("Error al recuperar los registros")
        finally:
            cursor.close()
            connection.close()    

    def create_index_psql(self, schema, query):
        try:
            sqlStatement_update = ""

            if isinstance(query['columns'], list):
                columns_index = ",".join(query['columns'])
                print("Creando indice para lista de columnas", query['columns'])
                sqlStatement_update = generate_update_query(schema,query['language'], query['table_name'], f"vector_{query['index_name']}", query['columns'])
            else:
                sqlStatement_update = f"update {schema}.{query['table_name']} set vector_{query['index_name']} = to_tsvector('{query['language']}', {query['columns']});"
            
            sqlStatement_alter = f"alter table {schema}.{query['table_name']} add column vector_{query['index_name']} tsvector;"
            sqlStatement_create = f"create index {query['index_name']} on {schema}.{query['table_name']} using {query['method']}(vector_{query['index_name']});"
            
            print("SQL Statement")
            print(sqlStatement_alter)
            print(sqlStatement_update)
            print(sqlStatement_create)

            connection = dbconnection(self.data_config)
            cursor = connection.cursor()
            cursor.execute(sqlStatement_alter)
            #connection.commit()
            cursor.execute(sqlStatement_update)
            #connection.commit()
            cursor.execute(sqlStatement_create)
            connection.commit()
            cursor.close()
            connection.close()
            return "Indice en Postgres creado correctamente"
            
        except Exception as e:
            print(f"Error al crear el indice: {e}")
            connection.rollback()
            cursor.close()
            connection.close()
            raise ManageExceptions(f"Error al crear el indice GIN: {e}")

    def create_index_own(self, file, query,folder_path):
        try:
            columnas = query['columns'] if isinstance(query['columns'], list) else [query['columns']]
            if query.get('table_name'):
                folder_path += f"/{query['table_name']}"
            else:
                folder_path = file.replace('.','_')
            print("New folder: ", folder_path)
            os.makedirs(folder_path, exist_ok=True)
            s = SPIMI(file,bloques_dir=f"{folder_path}/index_blocks", columnas=columnas, language=query['language'])  
            s.spimi_invert()
            print("Spimi culminado")
            return True
        except Exception as e:
            print(f"Error al crear el indice: {e}")
            raise ManageExceptions(f"Error al crear el indice propio: {e}")
        

    def search_psql(self, schema, query, index_name, table, limit):
        query_search = preprocesamientoQuery(query)
        sqlStatement= f"""
            SELECT *,
               ts_rank_cd(vector_{index_name}, {query_search}) as score
                FROM {schema}.{table}, to_tsquery({query_search}) query
                WHERE vector_{index_name} @@ query
                ORDER BY score DESC
                LIMIT {limit};
		
        """
        try:
            connection = dbconnection(self.data_config)
            cursor = connection.cursor()
            cursor.execute(sqlStatement)
            result = cursor.fetchall()
            cursor.close()
            connection.close()

            column_names = [desc[0] for desc in cursor.description]

            output = io.StringIO()
            writer = csv.writer(output, delimiter='%')
            writer.writerow(column_names)  
            writer.writerows(result)         
            csv_text = output.getvalue()
            output.close()
            print("Primeros 50 registros en formato CSV:")
            return csv_text
            
        except Exception as e:
            print(f"Error al crear la tabla: {e}")
            connection.rollback()
            cursor.close()
            connection.close()
            raise ManageExceptions(e)
        
    def search_spimi(self, file, query, columns, table, limit, language_p,folder_path):
        try:
            columnas = columns if isinstance(columns, list) else [columns]
            if ".csv" not in table:
                folder_path += f"/{table}"
            else:
                folder_path = file.replace('.','_')

            print("New folder: ", folder_path)
            os.makedirs(folder_path, exist_ok=True)
            s = SPIMI(file,bloques_dir=f"{folder_path}/index_blocks", columnas=columnas, language=language_p) 
            #s.spimi_invert()
            info_retrieved = s.retrieval(query,int(limit))
            result = getResultados(info_retrieved, file)
            print(result)
            print("Spimi culminado")
            return result
        except Exception as e:
            print(f"Error al crear el indice: {e}")
            raise ManageExceptions(f"Error al crear el indice propio: {e}")

    def create_dataset():
        return
    
    def search_knn_seq(self, query_t, k):
        try: 
            embeddings_folder = 'vectores'
            images_csv = 'fashion-dataset/images.csv'
            searcher = ImageSearcher(embeddings_folder, images_csv)
            
            # Suponiendo que tienes un índice de consulta
            query_index = 6  # Índice de la imagen de consulta
            if query_index < len(searcher.embeddings):
                query = searcher.embeddings[query_index]
                imagen_q_name = searcher.image_names[query_index]
                imagen_q = searcher.images_df[searcher.images_df['filename'] == imagen_q_name].iloc[0]
                print("La imagen de la query es:", imagen_q)
                print("Link de la imagen:", imagen_q['link'])

                # Realizar la búsqueda KNN y obtener el CSV como una cadena
                k = 5
                csv_result = knnSequentialSearch(query, k, searcher)
                
                if csv_result:
                    print("Resultados en formato CSV:")
                    print(csv_result)
                    return csv_result
            else:
                print(f"El índice de consulta {query_index} está fuera de rango.")
                raise ManageExceptions(f"El índice de consulta {query_index} está fuera de rango.")
        except Exception as e:
            print(f"Error en busqueda de imagenes Knn Sequential: {e}")
            raise ManageExceptions(f"Error en busqueda de imagenes Knn Sequential: {e}")
    
    

def main(): 
    query = '''
    USE "POSTGRES";
    CREATE DATASET FROM "file_path.csv";
    CREATE TABLE songs (
        track_id varchar(22),
        lyrics varchar(27698),
        track_name varchar(123),
        track_artist varchar(51),
        track_album_name varchar(151),
        playlist_name varchar(120),
        playlist_genre varchar(5),
        playlist_subgenre varchar(25),
        language varchar(2),
        info_vector tsvector
    ) FROM "file_path.csv";
    
    CREATE INDEX "content_idx" ON songs USING GIN(lyrics) AND LANGUAGE("spanish");
    '''
    
    query_json = process_query(query)
    print(query_json)

if __name__ == "__main__":
    main()