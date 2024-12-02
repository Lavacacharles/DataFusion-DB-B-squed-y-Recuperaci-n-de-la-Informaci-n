# DataFusion-DB-B-squeda-y-Recuperaci-n-de-la-Informaci-n
- ! Importante:
  Avance del indice invertido parte algortimica: index_spimi1, en el file: SpimiIntento3_funciona pero el merge esta raro_..., ya se pueen procesar consultas
 Y en mergeBlocks.py, el mergeo es correcto
- En la rama compiler se encuentra el front y el back listo para fusionar

## Extracción de características
- A diferencia de utilizar la version matricial luego aplanada de las imágenes para indexar y probar los índices de recuperación, optamos por el enfoque de utilizar descriptores locales.

![image](https://github.com/user-attachments/assets/c476d28e-770d-42a8-b4c3-0a8175d19f06)

- Resumiendo el proceso, en las capas de convolución analizar la imágenes por cuadrillas de 7 x 7, luego de 3 x 3 en las sigueintes 4 capas, además de esto se aplican técnicas de MaxPooling y AvgPooling, que reducen el ruido tomando el valor máximo en cada grilla de 3 x 3 y la complejidad mediante la redución del tamaño de la imágen procesada. Al final de todo se aplica un ajuste del vector representativo aplanado con 1000 parámetros para obtener las características relevantes de las imágenes, en nuestro caso la variación de ResNET 152 genera un vector representativo de 2048 características.
  
## Descriptores locales
- Los descriptores se definen como factores que tienen detalles en los objetos multimedia pero de pequeñas proporciones de la imágen, estos que permiten describir las imágenes y en comparaciones múltiples es mucho más robusto por prevalacer en la similitud de características granulares de las imágenes, la idea detrás de su efectividad es que se obtienen mediante procesos minuciosos de exploración de imágenes, como son las redes neuronales CNN, utilizamos el modelo pre entrenado Resnet 152 que tiene la siguiente arquitecutura:

- En la práctica se aplican técnicas eficientes con modelos pre entrenados como:
  
- SIFT (Scale-Invariant Feature Transform):
Basada en detectar puntos clave en diferentes escalas y orientaciones. Utiliza un histograma de gradientes orientados para describir la región local alrededor de cada punto clave, lo que la hace resistente a cambios de escala, rotación, iluminación y perspectiva.

- SURF (Speeded-Up Robust Features):
Similar a SIFT, pero más rápida. Usa aproximaciones rápidas de convoluciones basadas en el uso de "cajas de filtro" para detectar puntos clave y describirlos utilizando un histograma de gradientes distribuidos dentro de una región de interés.

- Deep Learning-Based Descriptors:
  - SuperPoint: Entrenado mediante aprendizaje no supervisado, combina detección de puntos clave y generación de descriptores en un modelo profundo. Es robusto frente a transformaciones complejas.
  - DELF (Deep Local Features): Descriptores extraídos de redes convolucionales preentrenadas, optimizados para aplicaciones específicas como la recuperación de imágenes o la correspondencia visual.

Utilizaremos `SuperPoint` para la representación de imágenes, indexado y en la recuperación identificaremos las imágenes e utilizaremos Resnet para la evaluación de la más parecida de imágenes de consulta:
  
## Maldición de la dimensionalidad
La maldicion de la alta dimensionalidad es un fenomeno que ocurre conforme se incrementan las dimensiones de los vectores caracteristicos. Hace referencia a que a mas dimensiones, mas esparsos parecen los datos, las distancias convergen a ser las máximas e indistinguibles en un espacio infinito(Norma infinito), de modo que los datos antes presuntamente cercanos empiezan a perder la cercanía entre estos y la distancia se homologa para todos los datos.

Para lidiar con problemas se opta por ténicas de reducción de la dimensionalidad considerando 2 factores principales, conservar las relaciones o las estructuras y distancias, dependiendo del problema, en machine learning suele priorizarse mantener las relaciones, un ejemplo es PCA y sus variaciones como SVD, que capturan la varianza de los datos y redimensionan los datos manteniendo la máxima separabilidad posible para mejorar los modelos. Por otro lado ténicas como Random Projections que mantienen las distancias y son más eficientes computacionalmente porque aplican algoritmos de orden lineal, y es la técnica utilizada en el presente informa para poder experimentar con diferentes niveles de dimensionalidad.

- (esta explicación es una experiencia pasada, agregar aquí los resultados de la experimentación)
- Hemos observado el problema de la alta dimensionalidad en nuestro Rtree: (al tener datos demasiado esparsos, las 'bounding boxes' resultan cada ves menos informativas, se solapan, etc). Esto hace que la performance del indice tienda mas y mas a lineal, como vemos en nuestra experimentacion. Una solucion obvia para el problema de la dimensionalidad es reducir las dimensiones, pero al hacer esto (vimos en la practica) que la exactitud de nuestra busqueda era considerablemente menor. Esto tiene sentido, porque a menos datos para tomar una descicion, mas probable es equivocarse.

Referencias:
- Pustokhin, D. A., Singh, P. K., Choudhary, P., & Gunasekaran, M. (2021). An effective deep residual network based class attention layer with bidirectional LSTM for diagnosis and classification of COVID-19. Journal of Applied Statistics. Recuperado de https://www.researchgate.net/publication/347170147

- Baeldung. (n.d.). k-Nearest Neighbors and High Dimensional Data. Recuperado de https://www.baeldung.com/cs/k-nearest-neighbors.

- Götz, M., Wenning, M., & Voss, S. (2010). Adaptive nearest neighbor search on large graphs. DBVIS Technical Reports. Recuperado de https://bib.dbvis.de/uploadedFiles/190.pdf.
- Hannibunny. (n.d.). Gaussian Filter and Derivatives of Gaussian. Recuperado de https://hannibunny.github.io/orbook/referenceSection.html#citation-23.
