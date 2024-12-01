# Instructions

## Conda

La versión de gpu también tiene la versión de cpu

```sh
conda create --name knn_high
conda activate knn_high
conda install python faiss-gpu=1.6.5
```

Tal vez se necesite cambiar la versión de python e instalar otros paquetes para el uso de la targeta gráfica

```sh
conda install python=3.8
conda install -c conda-forge  cudatoolkit=10.1 faiss-gpu=1.6.5
```

## Pip

Incluye la versión de cpu.

```sh
pip install faiss-gpu
```

## Files required

En este drive hay un índice ya listo para usar. Entrenado con 2048 características
[link](https://drive.google.com/drive/folders/1Y_iCgfZCSZ6IQ9Rw1DSpxZuRUiWXMqEI?usp=sharing)
