import numpy as np
from rtree import index
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
import os
from typing import List, Tuple
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import gc
from sklearn.decomposition import PCA

class FashionImageSearch:
    def __init__(self, storage_type='disk', dimension_reduction=None):
        self.feature_extractor = None
        self.index = None
        self.storage_type = storage_type
        self.dimension_reduction = dimension_reduction
        self.pca = None
        
    def initialize_model(self):
        print("Inicializando modelo InceptionV3...")
        self.feature_extractor = InceptionV3(weights='imagenet', 
                                           include_top=False, 
                                           pooling='avg')
        
    def batch_insert(self, features_batch, paths_batch):
        """Insert a batch of features into the index"""
        for i, (feature, path) in enumerate(zip(features_batch, paths_batch)):
            try:
                idx = len(self.file_paths)
                coords = list(feature) + list(feature)
                self.index.insert(idx, coords)
                
                if self.storage_type == 'disk':
                    self.features_file.create_dataset(str(idx), data=feature)
                
                self.file_paths.append(path)
            except Exception as e:
                print(f"Error insertando item {idx}: {str(e)}")
                continue
    
    def read_styles_csv(self, styles_csv_path: str, limit: int = None) -> pd.DataFrame:
        try:
            # Intentar leer también images.csv si existe
            images_csv_path = os.path.join(os.path.dirname(styles_csv_path), 'images.csv')
            if os.path.exists(images_csv_path):
                print("Leyendo images.csv...")
                images_df = pd.read_csv(images_csv_path)
                print(f"Encontradas {len(images_df)} entradas en images.csv")
            
            print("Leyendo styles.csv...")
            styles_df = pd.read_csv(styles_csv_path, 
                                  encoding='utf-8',
                                  on_bad_lines='skip')
            
            if limit:
                styles_df = styles_df.head(limit)
            
            if 'id' not in styles_df.columns:
                raise ValueError("El archivo CSV debe contener una columna 'id'")
            
            return styles_df
            
        except Exception as e:
            print(f"Error al leer CSV: {str(e)}")
            raise
    
    def extract_features(self, image_path: str) -> np.ndarray:
        try:
            img = image.load_img(image_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.inception_v3.preprocess_input(x)
            features = self.feature_extractor.predict(x, verbose=0)
            del x
            gc.collect()
            return features.flatten()
        except Exception as e:
            print(f"Error procesando imagen {image_path}: {str(e)}")
            return None

    def initialize_pca(self, features_list):
        """Inicializar y ajustar PCA con un conjunto de características"""
        if len(features_list) < 2:
            print("No hay suficientes muestras para PCA")
            return None
            
        print("Inicializando PCA...")
        features_array = np.array(features_list)
        
        # Calcular el número óptimo de componentes (90% de la varianza o menos que el número de muestras)
        n_components = min(len(features_list) - 1, 
                         self.dimension_reduction if self.dimension_reduction else 256)
        
        print(f"Usando {n_components} componentes para PCA")
        self.pca = PCA(n_components=n_components)
        transformed_features = self.pca.fit_transform(features_array)
        
        # Actualizar la dimensión real usada
        self.dimension_reduction = n_components
        return transformed_features

    def build_index(self, images_path: str, styles_csv_path: str, limit: int = None):
        if self.feature_extractor is None:
            self.initialize_model()
        
        print("Leyendo archivo styles.csv...")
        styles_df = self.read_styles_csv(styles_csv_path, limit)
        print(f"Procesando {len(styles_df)} imágenes...")
        
        # Recolectar características para PCA
        initial_features = []
        print("Recolectando características iniciales para PCA...")
        for idx, row in tqdm(styles_df.iterrows(), total=len(styles_df)):
            if len(initial_features) >= 200:
                break
                
            image_path = os.path.join(images_path, f"{row['id']}.jpg")
            if not os.path.exists(image_path):
                continue
                
            features = self.extract_features(image_path)
            if features is not None:
                initial_features.append(features)
        
        # Inicializar PCA y obtener la dimensionalidad real
        if initial_features:
            transformed_features = self.initialize_pca(initial_features)
            if transformed_features is None:
                print("No se pudo inicializar PCA, usando características originales")
                self.dimension_reduction = len(initial_features[0])
            del initial_features
            gc.collect()
        
        # Configurar R-tree con las dimensiones correctas
        p = index.Property()
        p.dimension = self.dimension_reduction
        p.buffering_capacity = 100
        p.pagesize = 4096
        p.leaf_capacity = 50
        p.fill_factor = 0.7
        
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
        
        # Procesar todas las imágenes
        self.file_paths = []
        features_batch = []
        paths_batch = []
        batch_size = 10
        
        for idx, row in tqdm(styles_df.iterrows(), total=len(styles_df)):
            image_path = os.path.join(images_path, f"{row['id']}.jpg")
            
            if not os.path.exists(image_path):
                continue
                
            features = self.extract_features(image_path)
            if features is None:
                continue
            
            # Reducir dimensionalidad si PCA está inicializado
            if self.pca is not None:
                features = self.pca.transform(features.reshape(1, -1))[0]
            
            features_batch.append(features)
            paths_batch.append(image_path)
            
            if len(features_batch) >= batch_size:
                try:
                    self.batch_insert(features_batch, paths_batch)
                except Exception as e:
                    print(f"Error en batch insert: {str(e)}")
                finally:
                    features_batch = []
                    paths_batch = []
                    gc.collect()
        
        if features_batch:
            try:
                self.batch_insert(features_batch, paths_batch)
            except Exception as e:
                print(f"Error en último batch: {str(e)}")
            
        print(f"Índice construido con {len(self.file_paths)} imágenes")

    def search_similar(self, query_image_path: str, k: int = 5):
        """Search for similar images"""
        if self.feature_extractor is None or self.index is None:
            raise ValueError("Primero debe construir el índice con build_index()")
        
        query_features = self.extract_features(query_image_path)
        if query_features is None:
            raise ValueError("No se pudo procesar la imagen de consulta")
        
        if self.pca is not None:
            query_features = self.pca.transform(query_features.reshape(1, -1))[0]
            
        query_coords = list(query_features) + list(query_features)
        
        try:
            nearest = list(self.index.nearest(coordinates=query_coords, num_results=k))
        except Exception as e:
            print(f"Error en búsqueda: {str(e)}")
            return []
        
        results = []
        for n in nearest:
            try:
                if self.storage_type == 'disk':
                    features = self.features_file[str(n)][:]
                else:
                    features = np.array(self.index.get_bounds(n))[:self.dimension_reduction]
                
                distance = np.linalg.norm(features - query_features)
                results.append((self.file_paths[n], distance))
            except Exception as e:
                print(f"Error procesando resultado {n}: {str(e)}")
                continue
        
        return sorted(results, key=lambda x: x[1])
    
    def visualize_results(self, query_path: str, results: List[Tuple[str, float]]):
        """Visualize the search results"""
        plt.figure(figsize=(15, 3))
        
        plt.subplot(1, len(results) + 1, 1)
        query_img = Image.open(query_path)
        plt.imshow(query_img)
        plt.title('Query Image')
        plt.axis('off')
        
        for i, (path, distance) in enumerate(results, 2):
            plt.subplot(1, len(results) + 1, i)
            img = Image.open(path)
            plt.imshow(img)
            plt.title(f'Distance: {distance:.2f}')
            plt.axis('off')
        
        plt.show()

def main():
    BASE_PATH = "./fashion-dataset"
    IMAGES_PATH = os.path.join(BASE_PATH, "images")
    STYLES_CSV_PATH = os.path.join(BASE_PATH, "styles.csv")
    
    if not os.path.exists(IMAGES_PATH):
        print(f"Error: No se encuentra el directorio de imágenes: {IMAGES_PATH}")
        return
    if not os.path.exists(STYLES_CSV_PATH):
        print(f"Error: No se encuentra el archivo CSV: {STYLES_CSV_PATH}")
        return
    
    print(f"Usando directorio de imágenes: {IMAGES_PATH}")
    print(f"Usando archivo CSV: {STYLES_CSV_PATH}")
    
    # Crear instancia sin especificar dimensión de reducción
    searcher = FashionImageSearch(storage_type='disk')
    
    try:
        print("Construyendo índice...")
        searcher.build_index(
            images_path=IMAGES_PATH,
            styles_csv_path=STYLES_CSV_PATH,
            limit=None
        )

        # Guardar las rutas de las imágenes
        with open('file_paths.txt', 'w') as f:
            for path in searcher.file_paths:
                f.write(f"{path}\n")
        
        # Buscar una imagen de ejemplo
        test_image = os.path.join(IMAGES_PATH, '1525.jpg')
        if not os.path.exists(test_image):
            # Buscar la primera imagen disponible
            for file in os.listdir(IMAGES_PATH):
                if file.endswith('.jpg'):
                    test_image = os.path.join(IMAGES_PATH, file)
                    break
        
        if test_image:
            print(f"Buscando imágenes similares a: {test_image}")
            results = searcher.search_similar(test_image, k=5)
            
            if results:
                print("Mostrando resultados...")
                searcher.visualize_results(test_image, results)
            else:
                print("No se encontraron resultados")
        else:
            print("No se encontró ninguna imagen de prueba")
            
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()

if __name__ == "__main__":
    main()