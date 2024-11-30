import numpy as np
from rtree import index
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import requests
from io import BytesIO
import gc
from typing import List, Tuple

class CloudFashionImageSearch:
    def __init__(self, storage_type='disk'):
        self.feature_extractor = None
        self.index = None
        self.storage_type = storage_type
        self.pca = None
        self.image_urls = {}  # Diccionario para almacenar id -> url
        
    def initialize_model(self):
        print("Inicializando modelo InceptionV3...")
        self.feature_extractor = InceptionV3(weights='imagenet', 
                                           include_top=False, 
                                           pooling='avg')
    
    def load_image_urls(self, images_csv_path: str):
        """Cargar las URLs de las imágenes desde el CSV"""
        print("Cargando URLs de imágenes...")
        df = pd.read_csv(images_csv_path)
        self.image_urls = dict(zip(df['filename'].apply(lambda x: x.split('.')[0]), df['link']))
        print(f"Cargadas {len(self.image_urls)} URLs")
        return df
    
    def extract_features_from_url(self, image_url: str) -> np.ndarray:
        """Extract features from an image URL"""
        try:
            # Descargar imagen
            response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(response.content))
            
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensionar y procesar
            img = img.resize((299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.inception_v3.preprocess_input(x)
            
            # Extraer características
            features = self.feature_extractor.predict(x, verbose=0)
            
            # Limpiar memoria
            del x, img
            gc.collect()
            
            return features.flatten()
        except Exception as e:
            print(f"Error procesando imagen URL {image_url}: {str(e)}")
            return None

    def build_index(self, images_csv_path: str, limit: int = None):
        """Build index from image URLs"""
        if self.feature_extractor is None:
            self.initialize_model()
        
        # Cargar URLs
        images_df = self.load_image_urls(images_csv_path)
        if limit:
            images_df = images_df.head(limit)
        
        # Configurar R-tree
        p = index.Property()
        p.dimension = 2048
        p.buffering_capacity = 100
        
        if self.storage_type == 'disk':
            for ext in ['.idx', '.dat']:
                if os.path.exists(f'fashion_rtree{ext}'):
                    os.remove(f'fashion_rtree{ext}')
            if os.path.exists('fashion_features.h5'):
                os.remove('fashion_features.h5')
            
            self.index = index.Index('fashion_rtree', properties=p)
            self.features_file = h5py.File('fashion_features.h5', 'w')
        else:
            self.index = index.Index(properties=p)
        
        # Procesar imágenes
        self.file_paths = []
        features_batch = []
        urls_batch = []
        batch_size = 10
        
        print(f"Procesando {len(images_df)} imágenes...")
        for idx, row in tqdm(images_df.iterrows(), total=len(images_df)):
            features = self.extract_features_from_url(row['link'])
            if features is None:
                continue
            
            features_batch.append(features)
            urls_batch.append(row['link'])
            
            if len(features_batch) >= batch_size:
                self.batch_insert(features_batch, urls_batch)
                features_batch = []
                urls_batch = []
                gc.collect()
        
        if features_batch:
            self.batch_insert(features_batch, urls_batch)
        
        # Guardar URLs
        with open('image_urls.txt', 'w') as f:
            for url in self.file_paths:
                f.write(f"{url}\n")
        
        print(f"Índice construido con {len(self.file_paths)} imágenes")
    
    def batch_insert(self, features_batch, urls_batch):
        """Insert a batch of features into the index"""
        for i, (feature, url) in enumerate(zip(features_batch, urls_batch)):
            try:
                idx = len(self.file_paths)
                coords = list(feature) + list(feature)
                self.index.insert(idx, coords)
                
                if self.storage_type == 'disk':
                    self.features_file.create_dataset(str(idx), data=feature)
                
                self.file_paths.append(url)
            except Exception as e:
                print(f"Error insertando item {idx}: {str(e)}")
                continue

    def search_similar(self, query_url: str, k: int = 5):
        """Search for similar images using URL"""
        if self.feature_extractor is None or self.index is None:
            raise ValueError("Primero debe construir el índice")
        
        query_features = self.extract_features_from_url(query_url)
        if query_features is None:
            raise ValueError("No se pudo procesar la imagen de consulta")
            
        query_coords = list(query_features) + list(query_features)
        nearest = list(self.index.nearest(coordinates=query_coords, num_results=k))
        
        results = []
        for n in nearest:
            try:
                if self.storage_type == 'disk':
                    features = self.features_file[str(n)][:]
                else:
                    features = np.array(self.index.get_bounds(n))[:2048]
                
                distance = np.linalg.norm(features - query_features)
                results.append((self.file_paths[n], distance))
            except Exception as e:
                print(f"Error procesando resultado {n}: {str(e)}")
                continue
        
        return sorted(results, key=lambda x: x[1])

    def visualize_results(self, query_url: str, results: List[Tuple[str, float]]):
        """Visualize results using URLs"""
        plt.figure(figsize=(15, 3))
        
        # Mostrar imagen de consulta
        plt.subplot(1, len(results) + 1, 1)
        response = requests.get(query_url)
        query_img = Image.open(BytesIO(response.content))
        plt.imshow(query_img)
        plt.title('Query Image')
        plt.axis('off')
        
        # Mostrar resultados
        for i, (url, distance) in enumerate(results, 2):
            plt.subplot(1, len(results) + 1, i)
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            plt.imshow(img)
            plt.title(f'Distance: {distance:.2f}')
            plt.axis('off')
        
        plt.show()

def main():
    # Configuración
    IMAGES_CSV_PATH = "./fashion-dataset/images.csv"
    
    if not os.path.exists(IMAGES_CSV_PATH):
        print(f"Error: No se encuentra el archivo CSV: {IMAGES_CSV_PATH}")
        return
    
    # Crear instancia del buscador
    searcher = CloudFashionImageSearch(storage_type='disk')
    
    try:
        # Construir índice
        print("Construyendo índice...")
        searcher.build_index(
            images_csv_path=IMAGES_CSV_PATH,
            limit=None  # Ajusta según necesites
        )
        
        # Probar con una URL de ejemplo
        test_url = None
        with open('image_urls.txt', 'r') as f:
            test_url = f.readline().strip()
        
        if test_url:
            print(f"Buscando imágenes similares a: {test_url}")
            results = searcher.search_similar(test_url, k=5)
            
            if results:
                print("\nResultados encontrados:")
                for i, (url, distance) in enumerate(results, 1):
                    print(f"{i}. URL: {url}")
                    print(f"   Distancia: {distance:.4f}")
                
                print("\nMostrando resultados visualmente...")
                searcher.visualize_results(test_url, results)
            else:
                print("No se encontraron resultados")
                
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()