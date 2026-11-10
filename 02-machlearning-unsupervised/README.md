# Aprendizaje No Supervisado: Diagnóstico de Cáncer de Mama (Wisconsin Data)

Este repositorio contiene un análisis exhaustivo de Aprendizaje No Supervisado (ANS) aplicado al dataset **Breast Cancer Wisconsin (Diagnostic)**. El objetivo principal es explorar en qué medida los algoritmos de clustering pueden identificar la naturaleza maligna o benigna de una masa tumoral a partir de variables geométricas tomadas mediante Aspiración con Aguja Fina (FNA).

---

## 📄 Estructura del Repositorio

* `breast_cancer_unsupervised_learning.ipynb`: Notebook interactivo con la carga de datos, preprocesamiento, análisis exploratorio (EDA), reducción de dimensionalidad (PCA) y modelos de agrupación.
* `README.md`: Resumen ejecutivo del proyecto y documentación de uso.

---

## 🛠️ Contenido del Análisis

### 1. Preprocesamiento y PCA
* **Estandarización:** Se aplicó `StandardScaler` a las 30 características numéricas para neutralizar el impacto de las diferencias de escala.
* **Reducción de Dimensionalidad:** PCA proyectó los datos en 2 dimensiones manteniendo una varianza explicada representativa para su visualización bidimensional.

### 2. Algoritmos Evaluados
* **K-Means:** Evaluado con valores de $K \in [2, 6]$. Demostró el mejor desempeño global al separar las clases principales.
* **Clustering Jerárquico (Ward):** Se construyó el dendrograma para visualizar la jerarquía de separación celular.
* **DBSCAN:** Utilizado para detectar la presencia de densidad heterogénea y valores atípicos (ruido).

### 3. Principales Métricas e Indicadores
* **Silhouette Score:** Evaluación intrínseca de la cohesión y separación del clúster.
* **Adjusted Rand Index (ARI):** Evaluación extrínseca comparando las agrupaciones generadas con las etiquetas de diagnóstico reales (Maligno vs. Benigno).

---

## 🚀 Requisitos e Instalación

Para ejecutar este notebook en tu entorno local, asegúrate de tener instaladas las siguientes librerías de Python:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

Para iniciar el entorno Jupyter:

```bash
jupyter notebook breast_cancer_unsupervised_learning.ipynb
```

---

## 📌 Conclusiones Clave

1. **Agrupamiento Natural:** Existe una clara separación intrínseca entre perfiles citológicos benignos y malignos en el espacio de características.
2. **Relevancia Clínica:** Los modelos no supervisados sirven como herramientas efectivas para triage rápido o priorización de análisis histopatológicos en laboratorios con alta carga de trabajo.