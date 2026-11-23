Bank Prediction – Proyecto de Ciencia de Datos

1. Descripción del proyecto

Bank Prediction es un proyecto de Ciencia de Datos enfocado en el análisis y clasificación de información bancaria mediante técnicas de aprendizaje automático (Machine Learning).

El proyecto contempla un proceso de análisis exploratorio de datos (EDA), preparación y procesamiento de la información, entrenamiento de modelos de clasificación y evaluación de sus resultados.

El propósito es identificar patrones en el conjunto de datos y comparar el rendimiento de diferentes algoritmos para determinar su capacidad de clasificación.

2. Objetivo general

Desarrollar un proceso de análisis y clasificación de datos bancarios mediante algoritmos de aprendizaje automático, evaluando su rendimiento a través de diferentes métricas de clasificación.

3. Modelos implementados

El proyecto contempla los siguientes modelos:

- Baseline: modelo de referencia utilizado para establecer un punto de comparación con los demás algoritmos.
- MLP (Multi-Layer Perceptron): red neuronal artificial que permite identificar patrones y relaciones complejas en los datos.
- Naive Bayes: algoritmo probabilístico basado en el teorema de Bayes.
- SVM (Support Vector Machine): algoritmo que busca encontrar el límite óptimo de separación entre las clases.

4. Análisis exploratorio de datos (EDA)

Se realizó un análisis exploratorio de datos para conocer las características del conjunto de datos bancarios.

Entre las actividades desarrolladas se encuentran:

- Descripción general del conjunto de datos.
- Identificación de variables y tipos de datos.
- Análisis de valores faltantes.
- Distribución de las variables.
- Identificación de patrones y relaciones.
- Generación de gráficas para visualizar el comportamiento de los datos.

El análisis exploratorio se encuentra implementado en el archivo "notebooks/EDA.py".

5. Estructura del proyecto

bank-prediction/
│
├── data/
│   ├── draw/
│   │   └── Dataset original
│   │
│   ├── processed/
│   │   └── Datos procesados para entrenamiento y prueba
│   │
│   ├── make_dataset.py
│   └── preprocess.py
│
├── notebooks/
│   └── EDA.py
│
├── reports/
│   ├── figures/
│   │   ├── EDA/
│   │   └── resultados/
│   │
│   └── articulo_ieee.pdf
│
├── src/
│   ├── config.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── mlp.py
│   │   ├── naive_bayes.py
│   │   └── svm.py
│   │
│   └── visualization/
│       └── plots.py
│
├── requirements.txt
│
└── README.md

6. Tecnologías utilizadas

Las principales herramientas y tecnologías utilizadas en el desarrollo del proyecto son:

- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Visual Studio Code

7. Instalación y configuración

Para ejecutar el proyecto es necesario contar con Python 3.11 y las dependencias correspondientes.

7.1 Clonar el repositorio

git clone URL_DEL_REPOSITORIO

Ingresar al directorio del proyecto:

cd bank-prediction

7.2 Crear un entorno virtual

python -m venv venv

Activar el entorno virtual en Windows:

venv\Scripts\activate

7.3 Instalar las dependencias

pip install -r requirements.txt

8. Organización y ejecución del proyecto

El proyecto se encuentra organizado en diferentes directorios, cada uno con una función específica:

- data: contiene el conjunto de datos original, los datos procesados y los scripts encargados de su preparación y transformación.
- notebooks: contiene el análisis exploratorio de datos.
- src: contiene la configuración general, los modelos de clasificación y las herramientas de visualización.
- reports: almacena las gráficas generadas durante el análisis y la evaluación, así como el artículo científico del proyecto.

Para trabajar con el proyecto se recomienda seguir este orden:

1. Revisar el conjunto de datos original.
2. Ejecutar los procesos de preparación y preprocesamiento.
3. Realizar el análisis exploratorio de datos.
4. Ejecutar los modelos de clasificación.
5. Generar y consultar las gráficas de resultados.
6. Analizar las métricas obtenidas y comparar el rendimiento de los modelos.

9. Evaluación de los modelos

Para evaluar y comparar el rendimiento de los modelos se consideran las siguientes métricas:

- Accuracy: mide la proporción de predicciones correctas.
- Precision: mide la proporción de predicciones positivas que fueron correctas.
- Recall: mide la capacidad del modelo para identificar correctamente los casos positivos.
- F1-Score: representa el equilibrio entre Precision y Recall.
- Matriz de confusión: permite visualizar los aciertos y errores de clasificación.
- Curva ROC y AUC: permiten analizar la capacidad de discriminación de los modelos.

10. Resultados

Los resultados obtenidos durante el desarrollo del proyecto se presentan mediante métricas de evaluación y representaciones gráficas.

Las visualizaciones del análisis exploratorio y de los resultados de los modelos se encuentran organizadas en el directorio "reports/figures/".

Esta información permite comparar el comportamiento de los algoritmos implementados e identificar sus fortalezas y limitaciones.

11. Documentación

El proyecto incluye un artículo científico en formato IEEE, ubicado en el directorio "reports/", donde se documenta el desarrollo, metodología y resultados obtenidos.

12. Autor

Proyecto académico desarrollado como parte de la asignatura de Ciencia de Datos.

Nombre del proyecto: Bank Prediction

Lenguaje de programación: Python

Área: Ciencia de Datos y Aprendizaje Automático.