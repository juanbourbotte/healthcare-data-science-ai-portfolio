# Módulo IV: Procesamiento Digital de Imágenes e IA en Salud

Este módulo aborda los fundamentos del **Procesamiento Digital de Imágenes (PDI)** y su integración con algoritmos de **Machine Learning e Inteligencia Artificial** aplicados al ámbito biomédico. A lo largo de cuatro notebooks prácticos, se desarrolla el flujo completo de trabajo: desde la manipulación matricial de píxeles y la curación de datos visuales hasta la restauración, segmentación y clasificación automatizada de estudios médicos.

---

## 📌 Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Modalidades de Imágenes en Salud](#modalidades-de-imágenes-en-salud)
3. [Estructura del Módulo](#estructura-del-módulo)
4. [Detalle de los Notebooks](#detalle-de-los-notebooks)
    * [1. Fundamentos y Estructura de Imágenes Digitales](#1-fundamentos-y-estructura-de-imágenes-digitales)
    * [2. Histograma, Mejora y Umbralado](#2-histograma-mejora-y-umbralado)
    * [3. Ruido y Filtros Espaciales](#3-ruido-y-filtros-espaciales)
    * [4. Práctica Integral: Mejora, Segmentación e Introducción a ML](#4-práctica-integral-mejora-segmentación-e-introducción-a-ml)
5. [Requisitos e Instalación](#requisitos-e-instalación)
6. [Flujo de Trabajo Sugerido](#flujo-de-trabajo-sugerido)

---

## 📖 Descripción General

El análisis computacional de imágenes médicas requiere comprender tanto la naturaleza física y matricial de la captura como las técnicas de preprocesamiento necesarias para alimentar modelos predictivos.

En este módulo se desarrollan competencias para:
* Manipular imágenes digitales como arreglos multidimensionales respetando sus tipos de datos y rangos dinámicos.
* Realizar tareas de curación y preprocesamiento: mejora de contraste, restauración frente a distintos tipos de ruido y ajustes radiométricos o dimensionales.
* Implementar operadores espaciales lineales (convolución) y no lineales (estadísticos de orden).
* Combinar técnicas clásicas de segmentación con algoritmos de Aprendizaje Automático y Deep Learning para asistir en el diagnóstico clínico.

---

## 🏥 Modalidades de Imágenes en Salud

Los conceptos y pipelines del módulo se enmarcan en el procesamiento de diversas modalidades biomédicas:
* **Imágenes Ópticas (RGB):** Microscopía óptica y confocal, dermatología y fondo de ojo.
* **Imágenes Monocromáticas 2D:** Radiografías de tórax, mamografías, termografías y microscopía de barrido electrónico.
* **Estudios Volumétricos 3D y Multimodales:** Tomografías Computadas (CT), Resonancias Magnéticas (MRI) y estudios híbridos PET-CT.

---

## 📁 Estructura del Módulo

```plaintext
.
├── 01_Fundamentos_Imagenes_Digitales.ipynb
├── 02_Histograma_Mejora_Umbralado.ipynb
├── 03_Ruido_y_Filtros_Espaciales.ipynb
└── 04_Practica_Integral_Segmentacion_ML.ipynb
```

---

## 🔬 Detalle de los Notebooks

### 1. Fundamentos y Estructura de Imágenes Digitales (`01_Fundamentos_Imagenes_Digitales.ipynb`)
* **Objetivo:** Comprender la representación digital de una imagen y el impacto de su codificación numérica en el procesamiento.
* **Contenidos Clave:**
  * Estructura espacial: matrices 2D y tensores multicanal (filas, columnas y canales).
  * Tipos de datos y profundidad de bits: enteros sin signo (`uint8`), punto flotante (`float` para normalización y operaciones matemáticas sin pérdida de precisión) y arreglos booleanos (`logical` para máscaras).
  * Categorización según canales e intensidad: imágenes binarias, monocromáticas (escala de grises) y a color (espacio RGB).

---

### 2. Histograma, Mejora y Umbralado (`02_Histograma_Mejora_Umbralado.ipynb`)
* **Objetivo:** Analizar la distribución de intensidades y aplicar técnicas de realce visual y separación de objetos.
* **Contenidos Clave:**
  * Cálculo e interpretación del histograma para evaluar exposición, brillo y rango dinámico.
  * **Transformaciones punto a punto:** Ajuste lineal de contraste, negativo de imagen, corrección gamma y normalización radiométrica.
  * **Umbralado (*Thresholding*) y Binarización:** Selección de puntos de corte globales y adaptativos para aislar regiones de interés (ROI) frente al fondo.

---

### 3. Ruido y Filtros Espaciales (`03_Ruido_y_Filtros_Espaciales.ipynb`)
* **Objetivo:** Modelar degradaciones frecuentes en la adquisición de imágenes y aplicar operadores de vecindad para restauración y extracción de bordes.
* **Contenidos Clave:**
  * Noción de vecindad, desplazamiento de máscaras (*kernels*) y estrategias de manejo de bordes (*zero padding*, replicación y reflexión).
  * **Filtros Lineales (Invariantes al corrimiento):**
    * Filtro de valor medio (*box filter*) y **Filtro Gaussiano** (ponderado por distancia según $\sigma$) para suavizado y atenuación de ruido gaussiano.
    * Filtros de desplazamiento y filtros de realce (*sharpening*) para incrementar la nitidez de estructuras y bordes.
  * **Filtros Estadísticos / de Orden (No lineales):**
    * Ordenamiento de valores en vecindades locales (`ordfilt2`) mediante operadores de **mediana, mínimo y máximo**.
    * Eliminación de ruido impulsivo ("sal y pimienta") sin difuminar contornos mediante el filtro de mediana, y efectos morfológicos de expansión/erosión mediante filtros de máximo y mínimo.

---

### 4. Práctica Integral: Mejora, Segmentación e Introducción a ML (`04_Practica_Integral_Segmentacion_ML.ipynb`)
* **Objetivo:** Integrar las etapas de curación, segmentación y modelado predictivo en casos aplicados a imágenes biomédicas.
* **Contenidos Clave:**
  * Pipeline completo de preprocesamiento: reducción de ruido, binarización, limpieza morfológica y corregistro.
  * Extracción de características cuantitativas (área, intensidad media, morfología) a partir de regiones segmentadas.
  * **Introducción a IA y ML en Imágenes Médicas:**
    * *Clasificación:* Detección de patologías en radiografías (ej. neumonía/COVID-19), categorización de densidad mamaria (A, B, C, D) y clasificación de tumores cerebrales en MRI.
    * *Segmentación Semántica:* Identificación píxel a píxel de tejidos, tumores o placas arterioscleróticas mediante arquitecturas convolucionales como **U-Net**.
    * *Regresión Cuantitativa:* Estimación continua de parámetros clínicos como el diámetro aórtico en resonancia magnética mediante redes neuronales convolucionales (CNN).

---

## ⚙️ Requisitos e Instalación

Para ejecutar los notebooks de este módulo se requiere **Python 3.8+** y las principales librerías del ecosistema de visión artificial y aprendizaje automático:

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-image opencv-python scikit-learn jupyter
```

---

## 💡 Flujo de Trabajo Sugerido

1. Ejecutar secuencialmente los cuadernos `01` a `03` para consolidar la base matemática de matrices, histogramas y convolución espacial.
2. Comparar visualmente el comportamiento de los filtros lineales frente a los filtros de orden ante distintos tipos de ruido sintético y médico.
3. Abordar el cuaderno `04` para integrar el preprocesamiento con la extracción de métricas y modelos de clasificación supervisada.