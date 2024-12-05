import pickle
import numpy as np
from rtree import index
import umap
import requests
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms, models
import pandas as pd
import gc
import warnings
import os
from omegaconf import OmegaConf
from urllib.parse import urlparse  # Para manejo de URLs
warnings.filterwarnings('ignore')

class ImageFeatureExtractor:
    def __init__(self):
        # Configuración del modelo y preprocesamiento
        print("Configurando modelo ResNet...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet152(weights='DEFAULT')
        self.model.eval()
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model = self.model.to(self.device)
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_path_or_url):
        """
        Carga una imagen desde una ruta local o URL.
        """
        try:
            if os.path.exists(image_path_or_url):  # Imagen local
                image = Image.open(image_path_or_url).convert('RGB')
            else:  # Asumir que es una URL
                response = requests.get(image_path_or_url, timeout=10)
                response.raise_for_status()  # Lanza error si falla la descarga
                image = Image.open(BytesIO(response.content)).convert('RGB')

            print("Imagen cargada exitosamente.")
            return image
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            return None

    def extract_features(self, image_path_or_url):
        """
        Extrae características de una imagen dada su ruta o URL.
        """
        try:
            # Cargar imagen
            image = self.load_image(image_path_or_url)
            if image is None:
                return None

            # Preprocesar imagen
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Extraer características
            with torch.no_grad():
                features = self.model(image_tensor)

            # Aplanar el tensor a un vector
            feature_vector = features.squeeze().cpu().numpy()
            print(f"Características extraídas. Dimensiones: {feature_vector.shape}")
            return feature_vector
        except Exception as e:
            print(f"Error al extraer características: {e}")
            return None


class ImageSimilaritySearch:
    def __init__(self, embeddings_path, images_csv_path, index_dir="./saved_index", target_dims=128):
        self.target_dims = target_dims
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "rtree.idx")
        self.data_path = os.path.join(index_dir, "search_data.pkl")
        
        # Crear directorio si no existe
        if not os.path.exists(index_dir):
            print(f"Creando directorio {index_dir}...")
            os.makedirs(index_dir)
        
        print("Cargando mapping de imágenes...")
        self.image_df = pd.read_csv(images_csv_path)
        self.url_to_filename = {}
        self.filename_to_url = {}
        self.url_to_index = {}
        self.filename_to_index = {}
        
        # Crear mappings
        for index, row in self.image_df.iterrows():
            self.filename_to_url[row['filename']] = row['link']
            self.url_to_filename[row['link']] = row['filename']
        
        # Configurar el extractor de características
        self.setup_feature_extractor()
        
        # Verificar archivos del índice
        index_dat_exists = os.path.exists(self.index_path + '.dat')
        index_idx_exists = os.path.exists(self.index_path + '.idx')
        data_exists = os.path.exists(self.data_path)
        
        if index_dat_exists and index_idx_exists and data_exists:
            print("\nCargando índice existente...")
            try:
                self.load_index()
                print("Índice cargado exitosamente.")
            except Exception as e:
                print(f"\nError al cargar el índice: {str(e)}")
                print("Creando nuevo índice...")
                self.create_new_index(embeddings_path)
        else:
            print("\nCreando nuevo índice...")
            self.create_new_index(embeddings_path)

    def setup_resnet_extractor(self):
        """Configura ResNet para extracción de características."""
        print("Configurando extractor ResNet...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargar configuraciones de OmegaConf
        resnet_config = OmegaConf.create({
            "model_name": "resnet152",
            "pool": "avgpool",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        })
        
        # Inicializar el modelo ResNet
        self.resnet_model = models.resnet152(weights='DEFAULT')
        self.resnet_model.eval()
        self.resnet_model = self.resnet_model.to(self.device)
        
        # Preprocesamiento de imágenes
        self.preprocess_resnet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup_feature_extractor(self):
        """Configura ResNet para extracción de características."""
        print("Configurando extractor ResNet...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Inicializar el modelo ResNet
        self.model = models.resnet152(weights='DEFAULT')
        self.model.eval()
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))  # Eliminar capa FC
        self.model = self.model.to(self.device)
        
        # Preprocesamiento de imágenes
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features_resnet(self, image_path):
        """Extrae el vector característico de una imagen usando ResNet."""
        if urlparse(image_path).scheme in ('http', 'https'):  # Si es URL, descargar la imagen
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:  # Si es archivo local
            img = Image.open(image_path).convert('RGB')
        
        # Preprocesar y mover a dispositivo
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Extraer características
        with torch.no_grad():
            features = self.model(img_tensor)
        
        # Aplanar el tensor a un vector
        feature_vector = features.squeeze().cpu().numpy()
        print(f"Características extraídas. Dimensiones: {feature_vector.shape}")
        return feature_vector

    def extract_features_from_url(self, image_url):
        """Extrae características de una imagen desde una URL"""
        try:
            # Verificar que la URL sea válida
            if not image_url.startswith(('http://', 'https://')):
                print(f"URL inválida: {image_url}")
                return None

            # Descargar imagen con manejo de errores HTTP
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()  # Lanza una excepción para códigos de estado HTTP no exitosos

            # Verificar el tipo de contenido
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                print(f"El contenido no es una imagen. Tipo de contenido: {content_type}")
                return None

            # Guardar el contenido en un buffer y verificar que sea una imagen válida
            image_data = BytesIO(response.content)
            try:
                image = Image.open(image_data)
                # Verificar que la imagen se puede leer completamente
                image.load()
            except Exception as e:
                print(f"Error al abrir la imagen: {str(e)}")
                return None

            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocesar imagen
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Extraer características
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Convertir a numpy y aplanar
            features = features.squeeze().cpu().numpy()
            
            print(f"Características extraídas exitosamente. Forma: {features.shape}")
            return features
            
        except requests.exceptions.Timeout:
            print(f"Tiempo de espera agotado al intentar descargar la imagen: {image_url}")
        except requests.exceptions.RequestException as e:
            print(f"Error en la solicitud HTTP: {str(e)}")
        except Exception as e:
            print(f"Error al extraer características de la imagen: {str(e)}")
        return None

    def get_vector_from_url(self, url):
        """Obtiene vector de características para una URL."""
        # Primero intentar obtener del dataset original
        if url in self.url_to_index:
            idx = self.url_to_index[url]
            print(f"Vector encontrado en el dataset (índice {idx})")
            return self.reduced_embeddings[idx]
        
        # Si no está, calcular dinámicamente
        print("Imagen no encontrada en el dataset, extrayendo características...")
        try:
            return self.extract_features_resnet(url)
        except Exception as e:
            print(f"Error al extraer características: {str(e)}")
            return None

    def create_new_index(self, embeddings_path):
        """Crea un nuevo índice desde cero"""
        print("Cargando embeddings...")
        self.load_embeddings(embeddings_path)
        print("Reduciendo dimensiones...")
        self.reduce_dimensions()
        print("Construyendo índice...")
        self.build_rtree_index()
        print("Guardando índice...")
        self.save_index()
        
    def load_embeddings(self, embeddings_path):
        """Carga los embeddings y crea los mappings necesarios"""
        with open(embeddings_path, 'rb') as file:
            dataset_nested = pickle.load(file)
            
        self.image_names = []
        embeddings = []
        
        for idx, item in enumerate(dataset_nested):
            if len(item["embedding"]) == 0:
                continue
            
            filename = item["image_name"]
            self.image_names.append(filename)
            embeddings.append(item["embedding"][0])
            
            self.filename_to_index[filename] = len(embeddings) - 1
            if filename in self.filename_to_url:
                url = self.filename_to_url[filename]
                self.url_to_index[url] = len(embeddings) - 1
            
        self.original_embeddings = np.array(embeddings)

    def reduce_dimensions(self):
        """Reduce dimensionalidad preservando similitud coseno"""
        print(f"Reduciendo dimensionalidad de {self.original_embeddings.shape[1]} a {self.target_dims}...")
        
        self.reducer = umap.UMAP(
            n_components=self.target_dims,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            verbose=True
        )
        
        self.reduced_embeddings = self.reducer.fit_transform(self.original_embeddings)
        del self.original_embeddings
        gc.collect()

    def build_rtree_index(self):
        """Construye el índice R-tree"""
        print("Construyendo índice R-tree...")
        
        p = index.Property()
        p.dimension = self.target_dims
        p.buffering_capacity = 64
        p.near_minimum_overlap_factor = 32
        p.variant = index.RT_Star
        p.leaf_capacity = 1000
        p.index_capacity = 1000
        p.fill_factor = 0.7
        p.tight_mbr = True
        p.overwrite = True
        
        self.idx = index.Index(self.index_path, properties=p)
        
        batch_size = 1000
        total_vectors = len(self.reduced_embeddings)
        
        for i in range(0, total_vectors, batch_size):
            batch_end = min(i + batch_size, total_vectors)
            batch = self.reduced_embeddings[i:batch_end]
            
            for j, embedding in enumerate(batch):
                idx = i + j
                epsilon = 1e-5
                bbox = tuple(list(embedding - epsilon) + list(embedding + epsilon))
                self.idx.insert(idx, bbox)
            
            if (i + batch_size) % 10000 == 0:
                print(f"Procesados {i + batch_size} de {total_vectors} vectores...")

    def cosine_similarity(self, v1, v2):
        """Calcula la similitud de coseno entre dos vectores"""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)
    
    def find_similar_images(self, query_url, k=5):
        """Encuentra imágenes similares para una URL de imagen."""
        query_vector = self.get_vector_from_url(query_url)
        if query_vector is None:
            raise ValueError(f"No se pudo obtener el vector característico para la URL: {query_url}")
        
        # Reducir dimensiones si es necesario
        if query_vector.shape[0] != self.target_dims:
            print(f"Reduciendo dimensiones del vector de query...")
            query_vector = self.reducer.transform([query_vector])[0]
        
        epsilon = 1e-5
        bbox_query = tuple(list(query_vector - epsilon) + list(query_vector + epsilon))
        nearest = list(self.idx.nearest(bbox_query, k))
        
        # Calcular similitudes coseno exactas
        similarities = [
            (idx, self.cosine_similarity(query_vector, self.reduced_embeddings[idx]))
            for idx in nearest
        ]
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [
            {
                'rank': i + 1,
                'filename': self.image_names[idx],
                'url': self.filename_to_url.get(self.image_names[idx], 'URL no disponible'),
                'similarity': float(sim)
            }
            for i, (idx, sim) in enumerate(similarities[:k])
        ]
        return results

    def save_index(self):
        """Guarda el índice y datos necesarios"""
        data_to_save = {
            'image_names': self.image_names,
            'reducer': self.reducer,
            'reduced_embeddings': self.reduced_embeddings,
            'url_to_index': self.url_to_index,
            'filename_to_index': self.filename_to_index,
            'url_to_filename': self.url_to_filename,
            'filename_to_url': self.filename_to_url
        }
        
        with open(self.data_path, 'wb') as f:
            pickle.dump(data_to_save, f)
            
    def load_index(self):
        """Carga el índice y datos guardados"""
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.image_names = data['image_names']
        self.reducer = data['reducer']
        self.reduced_embeddings = data['reduced_embeddings']
        self.url_to_index = data['url_to_index']
        self.filename_to_index = data['filename_to_index']
        self.url_to_filename = data['url_to_filename']
        self.filename_to_url = data['filename_to_url']
        
        p = index.Property()
        p.dimension = self.target_dims
        p.buffering_capacity = 64
        p.near_minimum_overlap_factor = 32
        p.variant = index.RT_Star
        p.leaf_capacity = 1000
        p.index_capacity = 1000
        
        self.idx = index.Index(self.index_path, properties=p)

if __name__ == "__main__":
    embeddings_path = "the_last_one.pkl"
    images_csv_path = "images.csv"
    
    print("Inicializando sistema de búsqueda...")
    search_system = ImageSimilaritySearch(embeddings_path, images_csv_path)
    
    # Ejemplo de uso con una URL externa
    query_url = "https://res.cloudinary.com/ddfhfbsdo/image/upload/v1733245126/cld-sample-5.jpg"
    
    try:
        print("\nBuscando imágenes similares...")
        similar_images = search_system.find_similar_images(query_url, k=5)
        
        print("\nImágenes más similares:")
        for result in similar_images:
            print(f"Rank {result['rank']}: {result['filename']}")   
            print(f"URL: {result['url']}")
            print(f"Similitud: {result['similarity']:.4f}\n")
    except Exception as e:
        print(f"Error durante la búsqueda: {str(e)}")