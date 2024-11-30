# import numpy as np
# from rtree import index
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.preprocessing import image
# import os
# import h5py
# import matplotlib.pyplot as plt
# import pandas as pd
# from tqdm import tqdm
# import requests
# from io import BytesIO
# import gc
# from typing import List, Tuple
# import joblib

# class HybridFashionSearch:
#     def __init__(self, storage_type='disk'):
#         self.feature_extractor = None
#         self.index = None
#         self.storage_type = storage_type
#         self.url_mapping = {}
#         self.dimension = 256  # Reducir dimensionalidad
#         self.pca = None
        
#     def initialize_model(self):
#         print("Inicializando modelo InceptionV3...")
#         self.feature_extractor = InceptionV3(weights='imagenet', 
#                                            include_top=False, 
#                                            pooling='avg')
        
#         # Inicializar PCA
#         from sklearn.decomposition import PCA
#         self.pca = PCA(n_components=self.dimension)
#         print(f"Inicializando PCA con {self.dimension} componentes...")
    
#     def load_url_mapping(self, images_csv_path: str):
#         """Cargar mapeo de IDs a URLs desde images.csv"""
#         print("Cargando mapeo de URLs...")
#         df = pd.read_csv(images_csv_path)
#         # Crear diccionario de filename -> url
#         self.url_mapping = dict(zip(
#             df['filename'].apply(lambda x: x.split('.')[0]),  # ID sin extensión
#             df['link']
#         ))
#         print(f"Cargadas {len(self.url_mapping)} URLs")
        
#     def extract_features(self, image_path: str) -> np.ndarray:
#         """Extraer características de imagen local"""
#         try:
#             img = image.load_img(image_path, target_size=(299, 299))
#             x = image.img_to_array(img)
#             x = np.expand_dims(x, axis=0)
#             x = tf.keras.applications.inception_v3.preprocess_input(x)
#             features = self.feature_extractor.predict(x, verbose=0)
#             return features.flatten()
#         except Exception as e:
#             print(f"Error procesando imagen {image_path}: {str(e)}")
#             return None

#     def build_index(self, images_path: str, images_csv_path: str, limit: int = None):
#         """Construir índice usando imágenes locales pero guardando URLs"""
#         if self.feature_extractor is None:
#             self.initialize_model()
            
#         # Cargar mapeo de URLs
#         self.load_url_mapping(images_csv_path)
        
#         # Configurar R-tree con parámetros más conservadores
#         p = index.Property()
#         p.dimension = self.dimension  # Usar dimensionalidad reducida
#         p.buffering_capacity = 10
#         p.pagesize = 1024
#         p.leaf_capacity = 10
#         p.index_capacity = 10
#         p.fill_factor = 0.3
#         p.near_minimum_overlap_factor = 3
#         p.variant = index.RT_Star  # Usar variante R*-tree que es más robusta
        
#         if self.storage_type == 'disk':
#             for ext in ['.idx', '.dat']:
#                 if os.path.exists(f'fashion_rtree{ext}'):
#                     os.remove(f'fashion_rtree{ext}')
#             if os.path.exists('fashion_features.h5'):
#                 os.remove('fashion_features.h5')
            
#             self.index = index.Index('fashion_rtree', properties=p)
#             self.features_file = h5py.File('fashion_features.h5', 'w')
#         else:
#             self.index = index.Index(properties=p)
        
#         # Recolectar características iniciales para PCA
#         print("Recolectando características iniciales para PCA...")
#         initial_features = []
#         files_for_pca = list(os.listdir(images_path))[:1000]  # Usar primeras 1000 imágenes para PCA
        
#         for filename in tqdm(files_for_pca, desc="Preparando PCA"):
#             if not filename.endswith('.jpg'):
#                 continue
            
#             image_path = os.path.join(images_path, filename)
#             features = self.extract_features(image_path)
#             if features is not None:
#                 initial_features.append(features)
        
#         # Ajustar PCA
#         if initial_features:
#             print("Ajustando PCA...")
#             self.pca.fit(np.array(initial_features))
#             del initial_features
#             gc.collect()
        
#         # Procesar imágenes
#         self.file_paths = []
#         features_batch = []
#         ids_batch = []
#         batch_size = 20  # Reducir tamaño del batch
        
#         image_files = os.listdir(images_path)
#         if limit:
#             image_files = image_files[:limit]
            
#         print(f"Procesando {len(image_files)} imágenes...")
#         for filename in tqdm(image_files, desc="Procesando imágenes"):
#             if not filename.endswith('.jpg'):
#                 continue
                
#             image_id = filename.split('.')[0]
#             if image_id not in self.url_mapping:
#                 continue
                
#             image_path = os.path.join(images_path, filename)
#             features = self.extract_features(image_path)
            
#             if features is None:
#                 continue
                
#             # Reducir dimensionalidad
#             features_reduced = self.pca.transform(features.reshape(1, -1))[0]
            
#             features_batch.append(features_reduced)
#             ids_batch.append(image_id)
            
#             if len(features_batch) >= batch_size:
#                 try:
#                     self.batch_insert(features_batch, ids_batch)
#                 except Exception as e:
#                     print(f"Error en batch insert: {str(e)}")
#                     # Intentar insertar uno por uno si falla el batch
#                     for i, (feat, id_) in enumerate(zip(features_batch, ids_batch)):
#                         try:
#                             self.batch_insert([feat], [id_])
#                         except Exception as e2:
#                             print(f"Error insertando item individual {i}: {str(e2)}")
#                 finally:
#                     features_batch = []
#                     ids_batch = []
#                     gc.collect()
        
#         if features_batch:
#             try:
#                 self.batch_insert(features_batch, ids_batch)
#             except Exception as e:
#                 print(f"Error en último batch: {str(e)}")
#                 # Intentar insertar uno por uno
#                 for i, (feat, id_) in enumerate(zip(features_batch, ids_batch)):
#                     try:
#                         self.batch_insert([feat], [id_])
#                     except Exception as e2:
#                         print(f"Error insertando último item individual {i}: {str(e2)}")
            
#         # Guardar mapeo de URLs
#         with open('image_urls.txt', 'w') as f:
#             for url in self.file_paths:
#                 f.write(f"{url}\n")
        
#         # Guardar el modelo PCA
#         joblib.dump(self.pca, 'pca_model.pkl')
                
#         print(f"Índice construido con {len(self.file_paths)} imágenes")
        
#     def batch_insert(self, features_batch, ids_batch):
#         """Insertar batch de características y guardar URLs correspondientes"""
#         for i, (feature, image_id) in enumerate(zip(features_batch, ids_batch)):
#             try:
#                 idx = len(self.file_paths)
#                 # Asegurarse de que las coordenadas sean del tamaño correcto
#                 coords = list(feature) + list(feature)
#                 self.index.insert(idx, coords)
                
#                 if self.storage_type == 'disk':
#                     self.features_file.create_dataset(str(idx), data=feature, 
#                                                     compression="gzip", 
#                                                     compression_opts=9)
                
#                 self.file_paths.append(self.url_mapping[image_id])
#             except Exception as e:
#                 print(f"Error insertando item {idx}: {str(e)}")
#                 continue

# def main():
#     BASE_PATH = "./fashion-dataset"
#     IMAGES_PATH = os.path.join(BASE_PATH, "images")
#     IMAGES_CSV_PATH = os.path.join(BASE_PATH, "images.csv")
    
#     if not os.path.exists(IMAGES_PATH):
#         print(f"Error: No se encuentra el directorio de imágenes: {IMAGES_PATH}")
#         return
#     if not os.path.exists(IMAGES_CSV_PATH):
#         print(f"Error: No se encuentra el archivo CSV: {IMAGES_CSV_PATH}")
#         return
    
#     print(f"Usando directorio de imágenes: {IMAGES_PATH}")
#     print(f"Usando archivo CSV: {IMAGES_CSV_PATH}")
    
#     searcher = HybridFashionSearch(storage_type='disk')
    
#     try:
#         print("Construyendo índice...")
#         searcher.build_index(
#             images_path=IMAGES_PATH,
#             images_csv_path=IMAGES_CSV_PATH,
#             limit=1000  # Procesar en lotes más pequeños
#         )
            
#     except Exception as e:
#         print(f"Error durante la ejecución: {str(e)}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()


# import numpy as np
# from rtree import index
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.preprocessing import image
# import os
# import h5py
# import matplotlib.pyplot as plt
# import pandas as pd
# from tqdm import tqdm
# from sklearn.decomposition import PCA
# import joblib
# import json

# class RTreeIndexBuilder:
#     def __init__(self, dimension=256):
#         self.feature_extractor = None
#         self.pca = None
#         self.dimension = dimension
#         self.index_config = {
#             'dimension': dimension,
#             'buffering_capacity': 10,
#             'leaf_capacity': 10,
#             'index_capacity': 10,
#             'fill_factor': 0.3,
#             'near_minimum_overlap_factor': 3,
#             'pagesize': 4096,
#             'variant': 'rtstar'
#         }
        
#     def save_config(self, path='index_config.json'):
#         """Guardar configuración del índice"""
#         with open(path, 'w') as f:
#             json.dump(self.index_config, f)
    
#     def initialize_models(self):
#         print("Inicializando modelos...")
#         self.feature_extractor = InceptionV3(weights='imagenet', 
#                                            include_top=False, 
#                                            pooling='avg')
#         self.pca = PCA(n_components=self.dimension)
        
#     def create_rtree_index(self):
#         """Crear índice R-tree con configuración específica"""
#         p = index.Property()
#         p.dimension = self.index_config['dimension']
#         p.buffering_capacity = self.index_config['buffering_capacity']
#         p.leaf_capacity = self.index_config['leaf_capacity']
#         p.index_capacity = self.index_config['index_capacity']
#         p.fill_factor = self.index_config['fill_factor']
#         p.near_minimum_overlap_factor = self.index_config['near_minimum_overlap_factor']
#         p.pagesize = self.index_config['pagesize']
#         p.variant = index.RT_Star
        
#         # Limpiar archivos existentes
#         for ext in ['.idx', '.dat']:
#             if os.path.exists(f'fashion_rtree{ext}'):
#                 os.remove(f'fashion_rtree{ext}')
                
#         return index.Index('fashion_rtree', properties=p)

#     def build_index(self, images_path: str, images_csv_path: str, limit: int = None):
#         """Construir índice R-tree"""
#         if self.feature_extractor is None:
#             self.initialize_models()
        
#         # Cargar mapeo de URLs
#         images_df = pd.read_csv(images_csv_path)
#         url_mapping = dict(zip(
#             images_df['filename'].apply(lambda x: x.split('.')[0]),
#             images_df['link']
#         ))
        
#         # Crear archivos de índice
#         rtree_idx = self.create_rtree_index()
#         features_file = h5py.File('fashion_features.h5', 'w')
        
#         # Recolectar características para PCA
#         print("Recolectando características para PCA...")
#         initial_features = []
#         files_sample = list(os.listdir(images_path))[:1000]  # Muestra para PCA
        
#         for filename in tqdm(files_sample):
#             if not filename.endswith('.jpg'):
#                 continue
                
#             image_path = os.path.join(images_path, filename)
#             img = image.load_img(image_path, target_size=(299, 299))
#             x = image.img_to_array(img)
#             x = np.expand_dims(x, axis=0)
#             x = tf.keras.applications.inception_v3.preprocess_input(x)
#             features = self.feature_extractor.predict(x, verbose=0)
#             initial_features.append(features.flatten())
        
#         # Ajustar PCA
#         print("Ajustando PCA...")
#         self.pca.fit(np.array(initial_features))
#         joblib.dump(self.pca, 'pca_model.pkl')
        
#         # Procesar todas las imágenes
#         file_paths = []
#         image_files = os.listdir(images_path)
#         if limit:
#             image_files = image_files[:limit]
            
#         print(f"Procesando {len(image_files)} imágenes...")
#         for idx, filename in enumerate(tqdm(image_files)):
#             if not filename.endswith('.jpg'):
#                 continue
                
#             image_id = filename.split('.')[0]
#             if image_id not in url_mapping:
#                 continue
                
#             image_path = os.path.join(images_path, filename)
            
#             # Extraer y reducir características
#             img = image.load_img(image_path, target_size=(299, 299))
#             x = image.img_to_array(img)
#             x = np.expand_dims(x, axis=0)
#             x = tf.keras.applications.inception_v3.preprocess_input(x)
#             features = self.feature_extractor.predict(x, verbose=0)
#             features_reduced = self.pca.transform(features.flatten().reshape(1, -1))[0]
            
#             # Insertar en R-tree
#             coords = list(features_reduced) + list(features_reduced)
#             rtree_idx.insert(idx, coords)
            
#             # Guardar características y URL
#             features_file.create_dataset(str(idx), data=features_reduced)
#             file_paths.append(url_mapping[image_id])
            
#             if idx % 100 == 0:
#                 rtree_idx.flush()  # Forzar escritura a disco
        
#         # Guardar URLs y configuración
#         with open('image_urls.txt', 'w') as f:
#             for url in file_paths:
#                 f.write(f"{url}\n")
                
#         self.save_config()
#         features_file.close()
        
#         print(f"Índice construido con {len(file_paths)} imágenes")

# def main():
#     BASE_PATH = "./fashion-dataset"
#     IMAGES_PATH = os.path.join(BASE_PATH, "images")
#     IMAGES_CSV_PATH = os.path.join(BASE_PATH, "images.csv")
    
#     if not all(os.path.exists(p) for p in [IMAGES_PATH, IMAGES_CSV_PATH]):
#         print("Error: No se encuentran los archivos necesarios")
#         return
    
#     builder = RTreeIndexBuilder(dimension=256)
    
#     try:
#         print("Construyendo índice...")
#         builder.build_index(
#             images_path=IMAGES_PATH,
#             images_csv_path=IMAGES_CSV_PATH,
#             limit=1000  # Ajusta según necesites
#         )
#     except Exception as e:
#         print(f"Error durante la construcción: {str(e)}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()

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
from sklearn.decomposition import PCA
import joblib
import json
import gc

class RTreeIndexBuilder:
    def __init__(self, dimension=32):
        self.feature_extractor = None
        self.pca = None
        self.dimension = dimension
        self.index_config = {
            'dimension': dimension,
            'buffering_capacity': 100,
            'leaf_capacity': 50,
            'index_capacity': 100,
            'fill_factor': 0.7,
            'pagesize': 4096,
            'variant': 'rtstar'
        }
        
    def save_config(self, path='index_config.json'):
        """Guardar configuración del índice"""
        with open(path, 'w') as f:
            json.dump(self.index_config, f)
            
    def initialize_models(self):
        """Inicializar los modelos necesarios"""
        print("Inicializando modelos...")
        self.feature_extractor = InceptionV3(weights='imagenet', 
                                           include_top=False, 
                                           pooling='avg')
        print("Modelo InceptionV3 cargado")
        
    def create_rtree_index(self):
        """Crear índice R-tree con configuración optimizada"""
        p = index.Property()
        p.dimension = self.index_config['dimension']
        p.buffering_capacity = self.index_config['buffering_capacity']
        p.leaf_capacity = self.index_config['leaf_capacity']
        p.index_capacity = self.index_config['index_capacity']
        p.fill_factor = self.index_config['fill_factor']
        p.pagesize = self.index_config['pagesize']
        p.variant = index.RT_Star
        
        # Limpiar archivos existentes
        for ext in ['.idx', '.dat']:
            if os.path.exists(f'fashion_rtree{ext}'):
                os.remove(f'fashion_rtree{ext}')
                
        return index.Index('fashion_rtree', properties=p)

    def build_index(self, images_path: str, images_csv_path: str, limit: int = None):
        """Construir índice R-tree"""
        try:
            # Inicializar modelos
            if self.feature_extractor is None:
                self.initialize_models()
            
            # Cargar mapeo de URLs
            print("Cargando mapeo de URLs...")
            images_df = pd.read_csv(images_csv_path)
            url_mapping = dict(zip(
                images_df['filename'].apply(lambda x: x.split('.')[0]),
                images_df['link']
            ))
            
            # Recolectar características para PCA
            print("Recolectando características para PCA...")
            initial_features = []
            image_files = os.listdir(images_path)
            if limit:
                image_files = image_files[:limit]
            
            # Usar más muestras para PCA
            n_samples_pca = min(1000, len(image_files))
            
            # Recolectar features para PCA
            for filename in tqdm(image_files[:n_samples_pca], desc="Extrayendo features para PCA"):
                if not filename.endswith('.jpg'):
                    continue
                    
                image_path = os.path.join(images_path, filename)
                try:
                    img = image.load_img(image_path, target_size=(299, 299))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = tf.keras.applications.inception_v3.preprocess_input(x)
                    features = self.feature_extractor.predict(x, verbose=0)
                    initial_features.append(features.flatten())
                except Exception as e:
                    print(f"Error procesando imagen {image_path}: {str(e)}")
                    continue
                    
                if len(initial_features) % 100 == 0:
                    gc.collect()
            
            # Verificar y ajustar dimensiones PCA
            print("Configurando PCA...")
            initial_features_array = np.array(initial_features)
            n_samples, n_features = initial_features_array.shape
            max_components = min(self.dimension, n_samples, n_features)
            
            print(f"Forma de los datos: {initial_features_array.shape}")
            print(f"Usando {max_components} componentes PCA")
            
            self.pca = PCA(n_components=max_components)
            self.pca.fit(initial_features_array)
            
            # Liberar memoria
            del initial_features
            del initial_features_array
            gc.collect()
            
            # Guardar modelo PCA
            print("Guardando modelo PCA...")
            joblib.dump(self.pca, 'pca_model.pkl')

            # Crear archivos de índice
            rtree_idx = None
            features_file = None
            
            try:
                print("Creando índice R-tree...")
                rtree_idx = self.create_rtree_index()
                features_file = h5py.File('fashion_features.h5', 'w')
                
                # Procesar todas las imágenes
                file_paths = []
                successful_count = 0
                error_count = 0
                
                print(f"Procesando {len(image_files)} imágenes...")
                for idx, filename in enumerate(tqdm(image_files, desc="Indexando imágenes")):
                    if not filename.endswith('.jpg'):
                        continue
                    
                    image_id = filename.split('.')[0]
                    if image_id not in url_mapping:
                        continue
                    
                    try:
                        # Procesar imagen
                        image_path = os.path.join(images_path, filename)
                        img = image.load_img(image_path, target_size=(299, 299))
                        x = image.img_to_array(img)
                        x = np.expand_dims(x, axis=0)
                        x = tf.keras.applications.inception_v3.preprocess_input(x)
                        features = self.feature_extractor.predict(x, verbose=0)
                        features_reduced = self.pca.transform(features.flatten().reshape(1, -1))[0]
                        
                        # Crear bounding box con valores seguros
                        coords = []
                        for val in features_reduced:
                            val = float(val)
                            if val >= 0:
                                coords.extend([val, val + 0.0001])
                            else:
                                coords.extend([val - 0.0001, val])
                        
                        # Insertar en índice
                        rtree_idx.insert(idx, coords)
                        features_file.create_dataset(str(idx), data=features_reduced)
                        file_paths.append(url_mapping[image_id])
                        successful_count += 1
                        
                        # Mantenimiento periódico
                        if idx % 100 == 0:
                            rtree_idx.flush()
                            gc.collect()
                            
                    except Exception as e:
                        print(f"Error procesando imagen {filename}: {str(e)}")
                        error_count += 1
                        continue
                
                # Guardar URLs y configuración
                print("Guardando configuración...")
                with open('image_urls.txt', 'w') as f:
                    for url in file_paths:
                        f.write(f"{url}\n")
                
                self.save_config()
                print(f"Índice construido exitosamente con {successful_count} imágenes")
                print(f"Errores durante el proceso: {error_count}")
                
            finally:
                if features_file is not None:
                    features_file.close()
                if rtree_idx is not None:
                    rtree_idx.close()
                    
        except Exception as e:
            print(f"Error durante la construcción: {str(e)}")
            raise

def main():
    BASE_PATH = "./fashion-dataset"
    IMAGES_PATH = os.path.join(BASE_PATH, "images")
    IMAGES_CSV_PATH = os.path.join(BASE_PATH, "images.csv")
    
    if not all(os.path.exists(p) for p in [IMAGES_PATH, IMAGES_CSV_PATH]):
        print("Error: No se encuentran los archivos necesarios")
        return
    
    builder = RTreeIndexBuilder(dimension=32)
    
    try:
        print("Construyendo índice...")
        builder.build_index(
            images_path=IMAGES_PATH,
            images_csv_path=IMAGES_CSV_PATH,
            limit=10000
        )
    except Exception as e:
        print(f"Error durante la construcción: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()