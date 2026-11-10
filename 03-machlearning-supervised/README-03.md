# Módulo: Aprendizaje Supervisado (Supervised Learning)

¡Bienvenido al módulo de **Aprendizaje Supervisado**! Este repositorio contiene una serie de notebooks teóricos y prácticos diseñados para llevarte desde los fundamentos del aprendizaje automático guiado por datos etiquetados hasta la implementación de modelos avanzados y su aplicación en un proyecto práctico.

---

## 📌 Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Estructura del Módulo](#estructura-del-módulo)
3. [Detalle de los Notebooks](#detalle-de-los-notebooks)
    * [1. Regresión Lineal](#1-regresión-lineal)
    * [2. Regresión Logística](#2-regresión-logística)
    * [3. Árboles de Decisión](#3-árboles-de-decisión)
    * [4. Modelos Avanzados](#4-modelos-avanzados)
    * [5. Proyecto Integrador / Aplicación Práctica](#5-proyecto-integrador--aplicación-práctica)
4. [Requisitos e Instalación](#requisitos-e-instalación)
5. [Flujo de Trabajo Sugerido](#flujo-de-trabajo-sugerido)

---

## 📖 Descripción General

El **Aprendizaje Supervisado** es una rama del Machine Learning donde los algoritmos aprenden a partir de un conjunto de datos etiquetado ($X, y$), buscando encontrar una función $f(X) \approx y$ que permita hacer predicciones precisas sobre datos nuevos e no observados.

En este módulo aprenderás a:
* Diferenciar y abordar problemas de **Regresión** (predicción de variables continuas) y **Clasificación** (predicción de etiquetas categóricas).
* Comprender las bases matemáticas, intuición y optimización de cada algoritmo.
* Evaluar y optimizar el rendimiento de los modelos utilizando métricas clave.
* Aplicar las mejores prácticas para evitar problemas como el *underfitting* (subajuste) y el *overfitting* (sobreajuste).

---

## 📁 Estructura del Módulo

```plaintext
.
├── 01_Regresion_Lineal.ipynb
├── 02_Regresion_Logistica.ipynb
├── 03_Arboles_de_Decision.ipynb
├── 04_Modelos_Avanzados.ipynb
└── 05_Aplicacion_Practica.ipynb
```

---

## 🔬 Detalle de los Notebooks

### 1. Regresión Lineal (`01_Regresion_Lineal.ipynb`)
* **Objetivo:** Introducir el modelado de variables continuas mediante relaciones lineales.
* **Contenidos Clave:**
  * Regresión Lineal Simple y Múltiple.
  * Función de costo: Error Cuadrático Medio ($MSE$).
  * Optimización mediante el algoritmo de **Gradiente Descendente** (Gradient Descent).
  * Interpretación de coeficientes y evaluación con métricas ($R^2$, $MAE$, $RMSE$).
  * Introducción a la Regularización: *Lasso* ($L_1$) y *Ridge* ($L_2$).

---

### 2. Regresión Logística (`02_Regresion_Logistica.ipynb`)
* **Objetivo:** Pasar del análisis de regresión al primer algoritmo fundamental de clasificación.
* **Contenidos Clave:**
  * Fundamentos de la clasificación binaria y multiclase.
  * La función Sigmoide (Logística) y límites de decisión (Decision Boundaries).
  * Función de pérdida: Entropía Cruzada / *Log Loss*.
  * Matriz de Confusión y Métricas de Evaluación: *Precision*, *Recall*, *F1-Score*, *ROC-AUC*.

---

### 3. Árboles de Decisión (`03_Arboles_de_Decision.ipynb`)
* **Objetivo:** Comprender modelos no lineales basados en reglas de partición jerárquicas.
* **Contenidos Clave:**
  * Estructura de un Árbol de Decisión: Nodos de decisión, ramas y hojas.
  * Criterios de división para Clasificación (Impureza de Gini, Entropía) y Regresión (Varianza).
  * Visualización e interpretación del árbol.
  * Control del sobreajuste (*Overfitting*): Hiperparámetros principales (`max_depth`, `min_samples_split`, `min_samples_leaf`) y poda (*Pruning*).

---

### 4. Modelos Avanzados (`04_Modelos_Avanzados.ipynb`)
* **Objetivo:** Explorar algoritmos de alto rendimiento y técnicas de ensamble (*Ensemble Learning*).
* **Contenidos Clave:**
  * **Métodos de Ensamble:**
    * *Bagging*: Random Forest.
    * *Boosting*: Gradient Boosting, XGBoost / LightGBM.
  * **Otros Algoritmos Clave:**
    * Máquinas de Vectores de Soporte (**SVM**): Margen máximo y *Kernel Trick*.
    * $k$-Vecinos más Cercanos (**k-NN**).
  * Optimización de Hiperparámetros: *GridSearchCV* y *RandomizedSearchCV*.
  * Validación Cruzada (*Cross-Validation*).

---

### 5. Proyecto Integrador / Aplicación Práctica (`05_Aplicacion_Practica.ipynb`)
* **Objetivo:** Aplicar los conceptos aprendidos a un caso de uso real de extremo a extremo (End-to-End).
* **Contenidos Clave:**
  * Exploración y Limpieza de Datos (EDA).
  * Preprocesamiento: Imputación de nulos, codificación de variables categóricas (*One-Hot Encoding*) y escalado de características.
  * Entrenamiento y comparación de múltiples modelos.
  * Selección del modelo óptimo basado en métricas del negocio.
  * Interpretación de importancia de variables (*Feature Importance*).

---

## ⚙️ Requisitos e Instalación

Para ejecutar los notebooks de este módulo, se recomienda contar con un entorno de **Python 3.8+**.

### Librerías necesarias:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm graphviz jupyter
```

Si prefieres ejecutar el proyecto directamente en la nube, puedes abrir el cuaderno práctico en **Google Colab** haciendo clic en el enlace dentro de `05_Aplicacion_Practica.ipynb`.

---

## 💡 Flujo de Trabajo Sugerido

1. **Revisa la teoría y práctica elemental** comenzando secuencialmente desde `01` hasta `04`.
2. **Experimenta** modificando hiperparámetros dentro de las celdas de código sugeridas en cada notebook.
3. **Ponte a prueba** completando los ejercicios integradores del cuaderno `05_Aplicacion_Practica.ipynb`.

---
*¡Mucho éxito en el estudio de este módulo! Si tienes dudas o sugerencias, siéntete libre de colaborar.*