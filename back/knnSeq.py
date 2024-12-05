import pickle
import numpy as np
import heapq
import os
import pandas as pd
from typing import List, Tuple
import io

class ImageSearcher:
    def __init__(self, embeddings_folder: str, images_csv: str):
        """
        Inicializa el buscador de imágenes cargando embeddings y datos de imágenes.

        :param embeddings_folder: Ruta a la carpeta que contiene archivos .pkl con embeddings.
        :param images_csv: Ruta al archivo CSV que contiene datos de imágenes.
        """
        self.embeddings, self.image_names = self.load_embeddings(embeddings_folder)
        self.images_df = self.load_image_data(images_csv)

    def load_embeddings(self, folder_path: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Carga los embeddings y los nombres de las imágenes desde archivos .pkl.

        :param folder_path: Ruta a la carpeta que contiene archivos .pkl.
        :return: Tuple con una lista de embeddings y una lista de nombres de imágenes.
        """
        archivos_pkl = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        embeddings = []
        image_names = []

        for archivo in archivos_pkl:
            archivo_pkl = os.path.join(folder_path, archivo)
            with open(archivo_pkl, 'rb') as f:
                datos = pickle.load(f)
            
            for item in datos:
                if 'embedding' in item and item['embedding'] is not None and len(item['embedding']) > 0:
                    embeddings.append(np.array(item['embedding'][0]))  # Asegura que sea un numpy array
                    image_names.append(item['image_name'])
        
        print(f"Cargados {len(embeddings)} embeddings de {len(archivos_pkl)} archivos.")
        return embeddings, image_names

    def load_image_data(self, csv_path: str) -> pd.DataFrame:
        """
        Carga los datos de imágenes desde un archivo CSV.

        :param csv_path: Ruta al archivo CSV.
        :return: DataFrame de pandas con los datos de imágenes.
        """
        images_df = pd.read_csv(csv_path)
        print(f"Cargado DataFrame con {len(images_df)} registros de imágenes.")
        return images_df

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calcula la similitud de coseno entre dos vectores.

        :param v1: Primer vector.
        :param v2: Segundo vector.
        :return: Similitud de coseno.
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def knn_search_cosine(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Realiza una búsqueda KNN utilizando la similitud de coseno.

        :param query: Vector de consulta.
        :param k: Número de vecinos más cercanos a encontrar.
        :return: Lista de tuplas con el nombre de la imagen y su similitud.
        """
        if len(self.embeddings) == 0:
            print("No hay embeddings cargados para realizar la búsqueda.")
            return []

        max_heap = []

        for i, data_vector in enumerate(self.embeddings):
            sim = self.cosine_similarity(data_vector, query)
            if len(max_heap) < k:
                heapq.heappush(max_heap, (sim, i))
            else:
                if sim > max_heap[0][0]:
                    heapq.heappushpop(max_heap, (sim, i))
        
        # Ordenar los resultados de mayor a menor similitud
        max_heap.sort(reverse=True)
        resultados = [(self.image_names[i], sim) for sim, i in max_heap]
        return resultados

    def get_image_link(self, image_name: str) -> str:
        """
        Obtiene el enlace de una imagen dado su nombre.

        :param image_name: Nombre del archivo de la imagen.
        :return: URL de la imagen o un mensaje indicando que no se encontró.
        """
        image_row = self.images_df[self.images_df['filename'] == image_name]
        if not image_row.empty:
            return image_row['link'].values[0]
        else:
            return "Enlace no encontrado."

def knnSequentialSearch(query: np.ndarray, k: int, searcher: ImageSearcher) -> str:
    """
    Realiza una búsqueda KNN secuencial y devuelve los resultados en formato CSV como una cadena.

    :param query: Vector de consulta.
    :param k: Número de vecinos más cercanos a encontrar.
    :param searcher: Instancia de ImageSearcher.
    :return: Cadena en formato CSV con los resultados.
    """
    print("Realizando búsqueda KNN...")
    resultados = searcher.knn_search_cosine(query, k)
    
    if not resultados:
        print("No se encontraron resultados.")
        return ""

    # Preparar los datos para el CSV
    csv_data = []
    for img_name, sim in resultados:
        link = searcher.get_image_link(img_name)
        csv_data.append({
            'Imagen': img_name,
            'Similitud': f"{sim:.4f}",
            'Link': link
        })
    
    # Crear un DataFrame
    df_resultados = pd.DataFrame(csv_data)
    
    # Convertir el DataFrame a una cadena CSV con delimitador %
    output = io.StringIO()
    df_resultados.to_csv(output, sep='%', index=False)
    csv_string = output.getvalue()
    output.close()
    
    return csv_string

if __name__ == "__main__":
    # Inicializar ImageSearcher
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
    else:
        print(f"El índice de consulta {query_index} está fuera de rango.")
