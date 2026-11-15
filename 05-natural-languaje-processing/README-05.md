# Módulo 5 (Parte 1): Procesamiento de Lenguaje Natural (NLP) y Minería de Textos en Salud

Gran parte de la información crítica en el ámbito sanitario se encuentra en formato no estructurado: evoluciones médicas, informes de alta, resultados de estudios por imágenes y respuestas abiertas en encuestas de satisfacción de pacientes. 

Abordaremos los fundamentos de la **Minería de Datos de Texto** y el **Procesamiento del Lenguaje Natural (NLP)** clásico y estadístico. Aprenderemos a limpiar, estructurar, analizar y vectorizar corpus textuales en español utilizando herramientas estándar de la industria, culminando con casos prácticos aplicados al análisis de textos clínicos y encuestas en salud.

> **Nota sobre la estructura del módulo:** Este módulo se divide en dos entregas:
> * **Parte 1 (Actual):** Fundamentos de NLP, preprocesamiento lingüístico, expresiones regulares, vectorización (TF-IDF, Embeddings), clustering y análisis de textos clínicos/encuestas.
> * **Parte 2 (Próxima entrega):** Introducción y manejo de Modelos Grandes del Lenguaje (LLMs).

---

## Objetivos de Aprendizaje

Al finalizar esta primera parte, podrás:
1. Comprender el ciclo de vida de un proyecto de minería de texto y los desafíos particulares del lenguaje clínico (abreviaturas, jerga médica, negaciones).
2. Utilizar **Expresiones Regulares (RegEx)** para detectar, extraer y anonimizar patrones específicos dentro de texto libre.
3. Construir *pipelines* de procesamiento lingüístico empleando **NLTK** y **spaCy** (tokenización, lematización, etiquetado gramatical y reconocimiento de entidades).
4. Realizar análisis exploratorio textual y transformar documentos en representaciones numéricas mediante **TF-IDF** y **Word Embeddings**.
5. Aplicar técnicas de **Word Clustering** para descubrir temáticas latentes en encuestas y notas clínicas.

---

## Contenidos del Módulo (Parte 1)

### 1. Minería de Datos de Texto e Introducción al NLP
* Diferencias entre datos estructurados y no estructurados en sistemas de información en salud.
* Flujo de trabajo estándar (*pipeline*) en NLP: ingesta, normalización, tokenización, representación y modelado.
* Particularidades y retos del procesamiento de texto en español y en dominios biomédicos.

### 2. Expresiones Regulares (RegEx)
* Sintaxis fundamental: metacaracteres, cuantificadores, clases de caracteres y grupos de captura.
* Aplicaciones prácticas: extracción de dosis, fechas, signos vitales, códigos diagnósticos y limpieza de ruido textual.

### 3. Herramientas y Bibliotecas (`NLTK` y `spaCy`)
* Arquitectura y casos de uso de **NLTK** (enfoque académico y modular) frente a **spaCy** (enfoque productivo basado en *pipelines*).
* Carga y uso de modelos preentrenados en español.

### 4. Preprocesamiento y Análisis Lingüístico
* **Limpieza y normalización:** manejo de mayúsculas, signos de puntuación, caracteres especiales y eliminación de palabras vacías (*stopwords*).
* **Tokenización:** segmentación de textos en oraciones y palabras (*tokens*).
* **Normalización morfológica:** *Stemming* vs. Lematización.
* **Etiquetado de Partes del Discurso (*POS Tagging*):** identificación de sustantivos, verbos, adjetivos y su rol sintáctico.
* **Extracción de Entidades Nombradas (*NER*):** detección automática de entidades relevantes (personas, lugares, organizaciones y conceptos clínicos como fármacos o patologías).

### 5. Análisis Exploratorio y Técnicas de Representación
* **Análisis Exploratorio de Textos (EDA):** conteo de frecuencias, análisis de *n-gramas* (bigramas, trigramas) y visualización mediante nubes de palabras.
* **TF-IDF (*Term Frequency - Inverse Document Frequency*):** ponderación estadística de términos relevantes frente a términos comunes del corpus.
* **Word Embeddings:** representaciones vectoriales densas que capturan similitud semántica y contextual (Word2Vec, FastText).
* **Word Clustering:** agrupamiento no supervisado de palabras y documentos para identificar patrones y tópicos frecuentes.

### 6. Aplicación Práctica: Textos Clínicos y Encuestas
* Procesamiento y estructuración de notas clínicas narrativas.
* Categorización y extracción de *insights* a partir de respuestas abiertas en encuestas de pacientes y personal de salud.

---

## Estructura del Material

* `presentacion_modulo5_parte1.pptx` *(actualizar con nombre exacto)*: Diapositivas teóricas con los conceptos clave, diagramas de flujo y ejemplos visuales.
* `notebook_modulo5_parte1.ipynb` *(actualizar con nombre exacto)*: Cuaderno interactivo en Python con ejemplos guiados paso a paso sobre textos clínicos y encuestas.
* `data/` *(si aplica)*: Conjunto de datos de prueba utilizados en las prácticas.

---

## Requisitos e Instalación

Para ejecutar el notebook de esta sección se recomienda contar con **Python 3.9+** y las siguientes bibliotecas:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk spacy gensim wordcloud
```

### Descarga de recursos lingüísticos necesarios

```bash
# Modelo en español para spaCy
python -m spacy download es_core_news_md
```

```python
# Recursos básicos de NLTK (ejecutar dentro de Python o del notebook)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

