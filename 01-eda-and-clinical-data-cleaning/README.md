# 📊 Módulo 1: Limpieza, Curación y Análisis Exploratorio de Datos Clínicos

## 🎯 Objetivos
- Implementar pipelines de curación de datos sobre registros clínicos heterogéneos.
- Resolver problemas de datos faltantes (*imputación*), valores atípicos (*outliers*) y consistencia de variables.
- Ejecutar un Análisis Exploratorio de Datos (EDA) guiado por hipótesis médicas/fisiológicas.

## 🔬 Metodología & Pasos Realizados
1. **Curación de Datos:** Corrección de formatos de fecha, tipado de variables categóricas/numéricas.
2. **Tratamiento de Nulos & Outliers:** Evaluación del mecanismo de pérdida (MCAR, MAR) e imputación por mediana/KNN. Filtro de outliers mediante Rango Intercuartílico (IQR).
3. **Análisis Estadístico:** Pruebas de normalidad (Shapiro-Wilk) y comparaciones inferenciales según la distribución del dataset.



# Análisis de Datos y Predicción de Diabetes - Pima Indians

Este proyecto tiene como objetivo analizar, limpiar e inferir patrones estadísticos sobre la presencia de diabetes en mujeres de la comunidad Pima, utilizando datos clínicos y modelos predictivos (regresión lineal y logística).

---

## 🗄️ Bases de Datos Utilizadas

El análisis contempla un total de **3 bases de datos** (o variantes estructuradas) del dataset de *Pima Indians Diabetes*:

1. **`pima-indians-diabetes.data.csv` (Dataset Original)**
   * **Fuente:** UCI Machine Learning Repository / GitHub (`jbrownlee`).
   * **Descripción:** Contiene los datos crudos en inglés de 768 pacientes femeninas de al menos 21 años de origen Pima. 
   * **Variables (9):** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, `Outcome`.

2. **`pima-indians-diabetes-espanol.csv` (Dataset Traducido)**
   * **Descripción:** Adaptación del dataset original con los nombres de las variables traducidos al español para facilitar la interpretación del análisis exploratorio y reportes.
   * **Variables (9):** `num_embarazos`, `glucosa`, `presion_sist`, `pliegue_tricip`, `insulina`, `bmi`, `anteced_dbt`, `edad`, `dbt`.

3. **`pima-indians-imputed.csv` (Dataset Limpio e Imputado)**
   * **Descripción:** Versión procesada donde los valores implícitos nulos (registrados originalmente como `0` en variables biológicamente imposibles de tener cero, como *glucosa*, *presión arterial*, *IMC*, *insulina* y *pliegue cutáneo*) fueron identificados e imputados utilizando la **mediana** de cada variable.
   * **Uso:** Base final utilizada para la modelación estadística y regresiones.

---

## 📊 Análisis y Visualización de la Pima Database

En los notebooks del proyecto se realizaron visualizaciones avanzadas e inferencias estadísticas sobre la base de datos Pima:

* **Distribución de Glucosa por Diagnóstico:**
  * Uso de histogramas y diagramas de caja (boxplots) interactivos (`Plotly`) para comparar la concentración de glucosa en sangre entre pacientes sanas (`0`) y diabéticas (`1`).
* **Relación IMC vs. Glucosa:**
  * Gráficos de dispersión (Scatter Plots) con líneas de tendencia lineal ($OLS$) para evaluar la correlación entre el Índice de Masa Corporal (IMC) y los niveles de glucosa.
* **Intervalos de Confianza:**
  * Cálculo y graficado de intervalos de confianza al 95% para los niveles medios de glucosa según el diagnóstico.
* **Inferencia Estadística:**
  * **Prueba T de Student:** Evaluación de diferencias estadísticamente significativas ($p < 0.05$) en la glucosa según la presencia de diabetes.
  * **Modelos de Regresión:** Implementación de Regresión Lineal Múltiple (predicción de glucosa) y Regresión Logística (cálculo de *Odds Ratios* y probabilidad de padecer diabetes en función de la edad, IMC y glucosa).

---

## 🚀 Requisitos e Instalación

Para ejecutar los notebooks de este proyecto, se requiere Python 3.x y las siguientes librerías:

```bash
pip install pandas numpy matplotlib seaborn plotly scipy statsmodels scikit-learn


## 📊 Práctico I: Análisis Exploratorio, Calidad de Datos e Inferencia (CODIES I)

Este módulo aborda el ciclo completo de curación, análisis exploratorio (EDA) y modelado inicial sobre la base epidemiológica **CODIES I** (Córdoba).

### 🎯 Objetivos y Alcance
- **Unidad de Observación:** Individuos adultos (>=18 años) participantes del estudio epidemiológico CODIES I.
- **Limpieza y Calidad de Datos:** Identificación de duplicados por ID, unificación de unidades de medida (estatura en metros), recodificación de inconsistencias en variables cualitativas y tratamiento de valores atípicos.
- **Ingeniería de Características:** Generación de variables categóricas clínicas (`edad_cat` y `colesterol_cat`).
- **Tratamiento de Missing Data:** Diagnóstico de valores nulos y análisis comparativo de impacto pre/post imputación por mediana en perfiles lipídicos.

### 📈 Metodología y Pruebas Estadísticas Aplicadas
1. **Análisis Descriptivo (EDA):** Caracterización de variables continuas (Media, Mediana, DE, IQR) y cualitativas (distribuciones de frecuencia y gráficos de proporciones).
2. **Evaluación de Inconsistencias:** Depuración de errores de carga (ej. corrección del valor anómalo `sexo=10` e hipercolesterolemia).
3. **Pruebas de Hipótesis e Inferencia:**
   - **Test Chi-Cuadrado ($\chi^2$):** Evaluación de independencia entre `sexo` y prevalencia de `diabetes`.
   - **$t$-Test / Mann-Whitney:** Comparación de promedios de `edad` según presencia/ausencia de `diabetes`.
   - **Correlación de Pearson y Spearman:** Evaluación de la asociación lineal y monótona entre `edad` y `colesterol`.
4. **Modelado Clásico:** Regresión Logística Múltiple para evaluar el riesgo de Diabetes (`dbt`), estimando Odds Ratios (OR) e Intervalos de Confianza (95%).

### 🛠️ Estructura del Archivo
- `Resolucion Practico I.ipynb`: Notebook ejecutable de Python con todo el pipeline de análisis, gráficos explicativos e interpretación clínica de p-valores y coeficientes.