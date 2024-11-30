import numpy as np
from rtree import index
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
import os
import h5py
import matplotlib.pyplot as plt
from typing import List, Tuple
import requests
from io import BytesIO
from sklearn.decomposition import PCA
import joblib
import json

class RTreeQueryEngine:
    def __init__(self):
        self.feature_extractor = None
        self.index = None
        self.file_paths = []
        self.features_file = None
        self.pca = None
        self.index_config = None
        
    def load_config(self, path='index_config.json'):
        """Cargar configuración del índice"""
        with open(path, 'r') as f:
            self.index_config = json.load(f)
            
    def initialize_model(self):
        print("Inicializando modelo InceptionV3...")
        self.feature_extractor = InceptionV3(weights='imagenet', 
                                           include_top=False, 
                                           pooling='avg')
    
    def load_index(self, index_dir: str = '.'):
        """Cargar índice pre-construido"""
        try:
            # Verificar archivos primero
            self.verify_files(index_dir)
            
            # Cargar configuración
            self.load_config()
            
            # Cargar modelo y PCA
            if self.feature_extractor is None:
                self.initialize_model()
            
            pca_path = os.path.join(index_dir, 'pca_model.pkl')
            print(f"Cargando PCA desde {pca_path}")
            self.pca = joblib.load(pca_path)
            
            # Cargar URLs
            urls_path = os.path.join(index_dir, 'image_urls.txt')
            print(f"Cargando URLs desde {urls_path}")
            with open(urls_path, 'r') as f:
                self.file_paths = [line.strip() for line in f]
            
            # Configurar R-tree
            p = index.Property()
            p.dimension = self.index_config['dimension']
            p.buffering_capacity = self.index_config['buffering_capacity']
            p.leaf_capacity = self.index_config['leaf_capacity']
            p.index_capacity = self.index_config['index_capacity']
            p.fill_factor = self.index_config['fill_factor']
            p.pagesize = self.index_config['pagesize']
            p.variant = index.RT_Star
            
            # Cargar R-tree
            rtree_path = os.path.join(index_dir, 'fashion_rtree')
            print(f"Cargando R-tree desde {rtree_path}")
            self.index = index.Index(rtree_path, properties=p)
            
            # Cargar features
            features_path = os.path.join(index_dir, 'fashion_features.h5')
            print(f"Cargando features desde {features_path}")
            self.features_file = h5py.File(features_path, 'r')
            
            print(f"Índice cargado con {len(self.file_paths)} imágenes")
            
        except Exception as e:
            print(f"Error detallado al cargar el índice: {str(e)}")
            raise

    def verify_files(self, index_dir):
        required_files = [
            'fashion_rtree.idx',
            'fashion_rtree.dat',
            'fashion_features.h5',
            'image_urls.txt',
            'pca_model.pkl',
            'index_config.json'
        ]
        
        missing = []
        for file in required_files:
            path = os.path.join(index_dir, file)
            if not os.path.exists(path):
                missing.append(file)
            else:
                size = os.path.getsize(path)
                print(f"Archivo {file} encontrado: {size} bytes")
                
        if missing:
            raise FileNotFoundError(f"Archivos faltantes: {', '.join(missing)}")
    
    def extract_features_from_url(self, image_url: str) -> np.ndarray:
        """Extraer características de imagen desde URL"""
        try:
            response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(response.content))
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize((299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.inception_v3.preprocess_input(x)
            features = self.feature_extractor.predict(x, verbose=0)
            features = features.flatten()
            
            # Aplicar PCA
            features = self.pca.transform(features.reshape(1, -1))[0]
            
            return features
        except Exception as e:
            print(f"Error procesando imagen URL {image_url}: {str(e)}")
            return None
    
    def search_similar(self, query_url: str, k: int = 5):
        """Buscar imágenes similares"""
        query_features = self.extract_features_from_url(query_url)
        if query_features is None:
            raise ValueError("No se pudo procesar la imagen de consulta")
        
        # Crear coordenadas de búsqueda usando el mismo enfoque que en el builder
        coords = []
        for val in query_features:
            val = float(val)
            if val >= 0:
                coords.extend([val, val + 0.0001])
            else:
                coords.extend([val - 0.0001, val])
        
        # Buscar los k vecinos más cercanos
        nearest = list(self.index.nearest(coordinates=coords, num_results=k*2))  # Pedir más resultados por si hay que filtrar
        
        results = []
        for n in nearest:
            features = self.features_file[str(n)][:]
            distance = np.linalg.norm(features - query_features)
            results.append((self.file_paths[n], distance))
        
        # Ordenar por distancia y tomar los k mejores
        return sorted(results, key=lambda x: x[1])[:k]
    
    def visualize_results(self, query_url: str, results: List[Tuple[str, float]]):
        """Visualizar resultados"""
        plt.figure(figsize=(20, 4))  # Hacer la figura más grande
        
        try:
            # Mostrar imagen de consulta
            plt.subplot(1, len(results) + 1, 1)
            response = requests.get(query_url, timeout=10)
            if response.status_code != 200:
                print(f"Error al cargar la imagen de consulta: {response.status_code}")
                return
                
            try:
                query_img = Image.open(BytesIO(response.content))
                plt.imshow(query_img)
                plt.title('Query Image', pad=20)
                plt.axis('off')
            except Exception as e:
                print(f"Error al abrir la imagen de consulta: {str(e)}")
                return
            
            # Mostrar resultados
            for i, (url, distance) in enumerate(results, 2):
                try:
                    plt.subplot(1, len(results) + 1, i)
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        plt.imshow(img)
                        plt.title(f'Distance: {distance:.4f}', pad=20)
                        plt.axis('off')
                    else:
                        print(f"Error al cargar imagen {url}: {response.status_code}")
                except Exception as e:
                    print(f"Error al procesar imagen {url}: {str(e)}")
                    continue
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error en visualización: {str(e)}")

def main():
    searcher = RTreeQueryEngine()
    
    try:
        print("Cargando índice...")
        searcher.load_index()
        
        test_url = "http://assets.myntassets.com/v1/images/style/properties/7a5b82d1372a7a5c6de67ae7a314fd91_images.jpg"
        
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
        print(f"Error durante la búsqueda: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()