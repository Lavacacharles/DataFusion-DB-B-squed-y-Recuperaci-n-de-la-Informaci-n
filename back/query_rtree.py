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

class FashionImageSearchQuery:
    def __init__(self):
        self.feature_extractor = None
        self.index = None
        self.file_paths = []
        self.features_file = None
        self.dimension = None
        
    def initialize_model(self):
        """Initialize the InceptionV3 model"""
        print("Inicializando modelo InceptionV3...")
        self.feature_extractor = InceptionV3(weights='imagenet', 
                                           include_top=False, 
                                           pooling='avg')
    
    def load_index(self, index_dir: str = '.'):
        """Load the pre-built index"""
        # Verificar que existan los archivos necesarios
        required_files = [
            os.path.join(index_dir, 'fashion_rtree.idx'),
            os.path.join(index_dir, 'fashion_rtree.dat'),
            os.path.join(index_dir, 'fashion_features.h5'),
            os.path.join(index_dir, 'file_paths.txt')
        ]
        
        for file in required_files[:3]:  # Verificar archivos principales
            if not os.path.exists(file):
                raise FileNotFoundError(f"No se encuentra el archivo de índice: {file}")
        
        # Cargar el modelo si no está inicializado
        if self.feature_extractor is None:
            self.initialize_model()
        
        # Cargar paths de imágenes
        if os.path.exists(required_files[3]):
            with open(required_files[3], 'r') as f:
                self.file_paths = [line.strip() for line in f]
        else:
            print("Advertencia: No se encontró el archivo de paths")
        
        # Cargar índice R-tree
        p = index.Property()
        p.dimension = 2048  # Dimensión original de InceptionV3
        self.index = index.Index(os.path.join(index_dir, 'fashion_rtree'))
        
        # Cargar archivo de características
        self.features_file = h5py.File(os.path.join(index_dir, 'fashion_features.h5'), 'r')
        
        print(f"Índice cargado con {len(self.file_paths)} imágenes")
        
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from an image"""
        try:
            img = image.load_img(image_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.inception_v3.preprocess_input(x)
            features = self.feature_extractor.predict(x, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"Error procesando imagen {image_path}: {str(e)}")
            return None
    
    def search_similar(self, query_image_path: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar images"""
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"No se encuentra la imagen: {query_image_path}")
            
        # Extraer características de la imagen de consulta
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            raise ValueError("No se pudo procesar la imagen de consulta")
        
        # Buscar los k vecinos más cercanos
        query_coords = list(query_features) + list(query_features)
        nearest = list(self.index.nearest(coordinates=query_coords, num_results=k))
        
        results = []
        for n in nearest:
            try:
                features = self.features_file[str(n)][:]
                distance = np.linalg.norm(features - query_features)
                results.append((self.file_paths[n], distance))
            except Exception as e:
                print(f"Error procesando resultado {n}: {str(e)}")
                continue
        
        return sorted(results, key=lambda x: x[1])
    
    def visualize_results(self, query_path: str, results: List[Tuple[str, float]]):
        """Visualize search results"""
        plt.figure(figsize=(15, 3))
        
        # Mostrar imagen de consulta
        plt.subplot(1, len(results) + 1, 1)
        query_img = Image.open(query_path)
        plt.imshow(query_img)
        plt.title('Query Image')
        plt.axis('off')
        
        # Mostrar resultados
        for i, (path, distance) in enumerate(results, 2):
            plt.subplot(1, len(results) + 1, i)
            img = Image.open(path)
            plt.imshow(img)
            plt.title(f'Distance: {distance:.2f}')
            plt.axis('off')
        
        plt.show()

def main():
    # Crear instancia del buscador
    searcher = FashionImageSearchQuery()
    
    try:
        # Cargar índice pre-construido
        print("Cargando índice...")
        searcher.load_index()
        
        # Buscar una imagen de ejemplo
        test_image = "./fashion-dataset/images/1163.jpg"  # Ajusta esta ruta
        
        print(f"Buscando imágenes similares a: {test_image}")
        results = searcher.search_similar(test_image, k=5)
        
        if results:
            print("\nResultados encontrados:")
            for i, (path, distance) in enumerate(results, 1):
                print(f"{i}. Archivo: {os.path.basename(path)}, Distancia: {distance:.4f}")
            
            print("\nMostrando resultados visualmente...")
            searcher.visualize_results(test_image, results)
        else:
            print("No se encontraron resultados")
            
    except Exception as e:
        print(f"Error durante la búsqueda: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()