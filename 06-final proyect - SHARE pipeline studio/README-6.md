# 🧠 Exploración y Predicción de Deterioro Cognitivo en Adultos Mayores (Estudio SHARE)

Pipeline integral de Ciencia de Datos e Inteligencia Artificial Aplicada a la Salud para la **segmentación fenotípica no supervisada** y **predicción de riesgo de deterioro cognitivo** en adultos mayores, utilizando datos transversales multicéntricos de la Ola 9 del estudio europeo **SHARE** (*Survey of Health, Ageing and Retirement in Europe*)[cite: 64, 70].

---

## 🎯 Problema Clínico y Objetivo
El deterioro cognitivo y la demencia suelen detectarse cuando el daño neurodegenerativo ya es clínicamente evidente, perdiendo una ventana crítica para intervenciones preventivas[cite: 64]. 

El objetivo de este proyecto es desarrollar un pipeline reproducible de Machine Learning capaz de:
1. **Explorar determinantes multidimensionales** (sociodemográficos, físicos, funcionales y afectivos) asociados a la salud cognitiva[cite: 70].
2. **Identificar fenotipos latentes de envejecimiento** mediante aprendizaje no supervisado[cite: 66, 70].
3. **Predecir el riesgo de bajo rendimiento cognitivo** optimizando métricas con sentido clínico para programas de *screening* poblacional[cite: 64, 66, 70].
4. **Auditar la explicabilidad del modelo** mediante valores SHAP para traducir las predicciones en umbrales clínicos accionables[cite: 66, 70].

---

## 📊 Dataset y Variable Objetivo (`SGC`)
* **Cohorte:** $N = 3,850$ participantes adultos mayores con 42 variables clínicas, funcionales, socioeconómicas y de personalidad[cite: 70].
* **Constructo Objetivo (`Y_deterioro`):** Derivado del **Score Global de Cognición (`SGC`, rango 0–29)**, que integra memoria verbal inmediata, memoria diferida, orientación temporoespacial y competencia numérica[cite: 65, 70]. Se definió como caso positivo (`Y = 1`, prevalencia 32.7%) a individuos con $\text{SGC} \le 8$[cite: 70].

---

## 🔬 Metodología y Pipeline Técnico

### 1. Análisis Exploratorio de Datos (EDA)
* Evaluación de normalidad mediante test de **Shapiro-Wilk**, confirmando distribuciones no paramétricas[cite: 70].
* Análisis de correlaciones bivariadas (**Spearman**) y multicolinealidad (**VIF**) entre dominios motores, funcionales y afectivos[cite: 70].
* Contrastes estadísticos (**Mann-Whitney U** y **Chi-cuadrado de Pearson**) por edad, sexo, nivel educativo, estado civil y país de residencia[cite: 70, 71].

### 2. Segmentación No Supervisada (Fenotipos de Envejecimiento)
* Preprocesamiento diferenciado: estandarización Z-score (`StandardScaler`) exclusiva para variables continuas y ordinales, conservando binarias y codificación *One-Hot* para nominales[cite: 67, 71].
* Evaluación comparativa mediante **Clustering Jerárquico (método de Ward)** y **K-Means**, auditados con Método del Codo, Coeficiente de Silueta, Índice Davies-Bouldin y proyección **PCA 2D**[cite: 68, 70].
* **Hallazgo Clínico ($K=2$):** Identificación de dos perfiles contrastantes[cite: 70]:
  * **Cluster 0 (*Envejecimiento Vulnerable*):** Edad media 80.9 años, SGC promedio 8.33, mayor carga de enfermedades crónicas (2.49), limitaciones motoras y sintomatología depresiva (EURO-D 5.29)[cite: 70].
  * **Cluster 1 (*Envejecimiento Activo*):** Edad media 71.7 años, SGC promedio 11.1, mayor reserva educativa (10.6 años) y mejor bienestar percibido (CASP 35.9)[cite: 70].

### 3. Modelado Supervisado y Calibración Clínica
* Partición estratificada **80% Train ($n=3,080$) / 20% Test ($n=770$)** sin *data leakage*[cite: 70].
* Optimización de hiperparámetros con `GridSearchCV` y **5-Fold Stratified Cross-Validation**, maximizando **AUPRC** debido al desbalance de clases[cite: 70].
* Selección de **Random Forest** regularizado (gap train/val = `0.066`, sin sobreajuste relevante)[cite: 70].

| Métrica en Test ($n=770$) | Umbral Estándar ($0.500$) | Umbral Optimizado Screening ($0.350$) |
| :--- | :---: | :---: |
| **ROC-AUC** | `0.761` | `0.761` |
| **AUPRC** *(baseline 0.327)* | `0.561` | `0.561` |
| **Recall / Sensibilidad** | `66.7%` | **`92.1%`** |
| **Especificidad** | `73.8%` | `44.4%` |
| **Precision (VPP)** | `55.3%` | `44.6%` |
| **F2-Score** *(prioriza Recall)* | `0.640` | **`0.759`** |

> **Impacto en Screening:** Ajustar el punto de corte a `0.350` permite detectar a **232 de los 252 pacientes reales con deterioro cognitivo** en la muestra de prueba, posicionando el modelo como una herramienta eficaz de tamizaje primario[cite: 70].

### 4. Explicabilidad Global y Local (SHAP)
* **Beeswarm Plot:** La **edad**, los **años de instrucción**, los **ingresos**, el **bienestar (`CASP`)**, las limitaciones instrumentales (`iadlza`) y la **fuerza muscular** lideran la capacidad predictiva[cite: 70].
* **Dependence Plots:** Detección de un **punto de quiebre no lineal entre los 73 y 74 años**, edad a partir de la cual el impacto marginal sobre el riesgo de deterioro se acelera abruptamente[cite: 70].

---

## 🛠️ Stack Tecnológico
`Python` • `Pandas` • `NumPy` • `Scikit-Learn` • `SciPy` • `Statsmodels` • `SHAP` • `Matplotlib` • `Seaborn` • `Plotly`

## 👥 Autores
Proyecto desarrollado en el marco de la **Diplomatura Universitaria en Ciencia de Datos e Inteligencia Artificial Aplicadas a la Investigación en Salud**[cite: 70]:
* **Juan Manuel Bourbotte Asensio**
* **Victoria De Los Milagros Lambert**
* **Maria Belen Boixados**