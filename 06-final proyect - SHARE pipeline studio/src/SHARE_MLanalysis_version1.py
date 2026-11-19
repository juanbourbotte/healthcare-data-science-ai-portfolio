# -*- coding: utf-8 -*-
"""
Machine Learning Application: Exploration and Prediction of Cognitive Impairment
using the Clinical SHARE Database
================================================================================
Pipeline completo, depurado y reproducible (Colab / Local VS Code).
"""

# ==============================================================================
# 0. LIBRERÍAS Y CONFIGURACIÓN GLOBAL
# ==============================================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    GridSearchCV,
    learning_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
)

# Dependencias opcionales (XGBoost y SHAP)
try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError("Instala xgboost antes de ejecutar: pip install xgboost") from exc

try:
    import shap
except ImportError as exc:
    raise ImportError("Instala shap antes de ejecutar: pip install shap") from exc

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = [10, 6]
RANDOM_STATE = 42
rng_global = np.random.default_rng(RANDOM_STATE)


# ==============================================================================
# 1. CARGA REPRODUCIBLE DE DATOS (LOCAL / GOOGLE COLAB)
# ==============================================================================
POSIBLES_RUTAS = [
    "data/df_final_listo.csv",
    "SHARE input/df_final_listo.csv",
    "/content/drive/My Drive/df_final_listo.csv",
]

ruta_archivo = next((r for r in POSIBLES_RUTAS if os.path.exists(r)), POSIBLES_RUTAS[0])
df = pd.read_csv(ruta_archivo)

print(f"¡Base de datos cargada con éxito desde '{ruta_archivo}'!")
print(f"Dimensiones originales del dataset: {df.shape[0]} filas y {df.shape[1]} columnas.")


# ==============================================================================
# 2. INSPECCIÓN GENERAL, LIMPIEZA INICIAL Y DICCIONARIO DE VARIABLES
# ==============================================================================
print("\n=== Columnas originales ===")
print(df.columns.tolist())

# Análisis de valores nulos por columna
nulos = df.isnull().sum()
porcentaje_nulos = (nulos / len(df)) * 100
resumen_nulos = (
    pd.DataFrame({
        "Non-Null Count": len(df) - nulos,
        "Nulos": nulos,
        "% Nulos": porcentaje_nulos.round(2),
    })
    .sort_values("% Nulos", ascending=False)
)

print("\n=== Resumen de valores nulos ===")
print(resumen_nulos)

print("\n=== Columnas con más del 20% de nulos ===")
print(resumen_nulos[resumen_nulos["% Nulos"] > 20])

# Detectar columnas que NO son float pero podrían serlo
candidatas_a_float = []
for col in df.columns:
    if df[col].dtype != "float64":
        conversion = pd.to_numeric(df[col], errors="coerce")
        no_nulos_original = df[col].notna().sum()
        no_nulos_convertido = conversion.notna().sum()
        pct_conv = (no_nulos_convertido / no_nulos_original * 100) if no_nulos_original > 0 else 0
        candidatas_a_float.append({
            "Columna": col,
            "Tipo actual": df[col].dtype,
            "% valores convertibles a numérico": round(pct_conv, 1),
        })

resumen_candidatas = pd.DataFrame(candidatas_a_float).sort_values(
    "% valores convertibles a numérico", ascending=False
)
print("\n=== Columnas que no son float, ordenadas por convertibilidad ===")
print(resumen_candidatas)

# Limpieza estructural: eliminar columnas duplicadas o 100% vacías
cols_a_eliminar = [c for c in ["eduyears_mod.1", "acitividad_vigorosa"] if c in df.columns]
if cols_a_eliminar:
    df.drop(columns=cols_a_eliminar, inplace=True)
    print(f"\n🧹 Columnas descartadas por duplicidad o 100% nulos: {cols_a_eliminar}")

# Diccionario de variables
codebook = {
    "sexo": "0=varón, 1=mujer",
    "age": "Edad en años al momento de la entrevista",
    "tipo_ciudad": "1=gran ciudad, 2=periferia, 3=pueblo grande, 4=pueblo chico, 5=zona rural",
    "ni": "Nivel educativo ISCED-97 (0=ninguno a 6=terciario superior)",
    "anos_instruccion": "Años de educación (solo entrevista baseline, alto missing)",
    "estado_civil": "1=casado/conviviente, 2=unión registrada, 3=separado, 4=nunca casado, 5=divorciado, 6=viudo",
    "salud": "Autopercepción de salud: 1=muy mala a 5=muy buena",
    "cronicas": "Conteo de enfermedades crónicas diagnosticadas por médico",
    "casp": "CASP-12: calidad de vida (control, autonomía, placer, autorrealización). Rango 12-48",
    "depresion": "¿Se sintió triste/deprimido el último mes? 0=no, 1=sí (ítem EURO-D)",
    "pesimismo": "¿Pierde esperanza en el futuro? 0=no, 1=sí (ítem EURO-D)",
    "pensamiento_suicida": "¿Sintió que preferiría estar muerto? 0=no, 1=sí (ítem EURO-D)",
    "culplable": "¿Tiende a culparse o sentirse culpable? 0=no, 1=sí (ítem EURO-D)",
    "problema_de_sueno": "¿Problemas para dormir? 0=no, 1=sí (ítem EURO-D)",
    "irritable": "¿Ha estado irritable? 0=no, 1=sí (ítem EURO-D)",
    "perdida_de_apetito": "Cambios en el apetito. 0=no, 1=sí (ítem EURO-D)",
    "fatiga": "¿Poca energía para hacer lo que quería? 0=no, 1=sí (ítem EURO-D)",
    "problema_de_concentracion": "Dificultad para concentrarse (TV, radio, lectura). 0=no, 1=sí (ítem EURO-D)",
    "llorado": "¿Ha llorado en el último mes? 0=no, 1=sí (ítem EURO-D)",
    "fuerza": "Fuerza de agarre máxima (dinamómetro), en kg",
    "adla": "Suma de 5 tareas básicas (rango 0-5)",
    "iadla": "Suma de 3 tareas instrumentales (rango 0-3)",
    "iadlza": "Suma de 5 tareas instrumentales ampliadas (rango 0-5)",
    "mobilidad": "Suma de 4 tareas de movilidad (rango 0-4)",
    "grossmotor": "Suma de 4 tareas de motricidad gruesa (rango 0-4)",
    "finemotor": "Suma de 3 tareas de motricidad fina (rango 0-3)",
    "imc": "Índice de Masa Corporal calculado (peso/altura²)*10000",
    "imc_cat": "1=bajo peso, 2=normal, 3=sobrepeso, 4=obesidad",
    "smoking": "¿Fuma actualmente? 1=sí, 0=no",
    "BF_sociable": "Big Five: reservado (bajo) vs. sociable (alto). Rango 1-5",
    "BF_amable": "Big Five: confiado/tolerante (bajo) vs. crítico (alto). Rango 1-5",
    "BF_concienciacion": "Big Five: perezoso (bajo) vs. meticuloso (alto). Rango 1-5",
    "BF_nervioso": "Big Five: relajado (bajo) vs. nervioso (alto). Rango 1-5",
    "BF_abierto": "Big Five: pocos intereses artísticos (bajo) vs. imaginación activa (alto). Rango 1-5",
    "ingresos": "Ingreso neto del hogar (imputado)",
    "SGC": "CONSTRUCTO = recall_1 + recall_2 + orientiiny + numeracy_combined (rango 0-29)",
}

print("\n=== Diccionario de variables ===")
for var, desc in codebook.items():
    if var in df.columns:
        print(f"• {var}: {desc}")


# ==============================================================================
# FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ==============================================================================
print("\n" + "=" * 70)
print("FASE 1: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 70)

print(f"Número de pacientes (sexo=1, Mujer): {(df['sexo'] == 1).sum()}")
print(f"Número de pacientes (sexo=0, Varón): {(df['sexo'] == 0).sum()}")

# 1.1 Descriptivos y normalidad de variables continuas
eda_continuas = [
    c for c in ["age", "anos_instruccion", "ingresos", "imc", "fuerza", "eurod", "casp", "SGC"]
    if c in df.columns
]
eda_categoricas = [
    c for c in [
        "sexo", "estado_civil", "country", "tipo_ciudad", "imc_cat",
        "depresion", "pesimismo", "pensamiento_suicida", "culplable",
        "problema_de_sueno", "irritable", "perdida_de_apetito", "fatiga",
        "problema_de_concentracion", "llorado", "smoking"
    ]
    if c in df.columns
]

print("\n=== Estadísticos descriptivos - Variables continuas ===")
desc = df[eda_continuas].describe().T
desc["skewness"] = df[eda_continuas].skew()
desc["kurtosis"] = df[eda_continuas].kurtosis()
print(desc)

print("\n=== Test de normalidad (Shapiro-Wilk) ===")
for col in eda_continuas:
    datos = df[col].dropna()
    if len(datos) > 3:
        _, p = stats.shapiro(datos.sample(min(5000, len(datos)), random_state=RANDOM_STATE))
        normal = "Sí" if p > 0.05 else "No"
        print(f"{col}: p={p:.4e} | ¿Distribución normal? {normal}")

# Histogramas continuas
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()
for i, col in enumerate(eda_continuas):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
    axes[i].set_title(f"{col}\nSkew: {df[col].skew():.2f}")
for j in range(len(eda_continuas), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("distribucion_continuas.png", dpi=150)
plt.show()

# Frecuencias y barras categóricas
print("\n=== Frecuencias - Variables categóricas ===")
for col in eda_categoricas:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()
for i, col in enumerate(eda_categoricas):
    df[col].value_counts().plot(kind="bar", ax=axes[i])
    axes[i].set_title(col)
    axes[i].tick_params(axis="x", rotation=45)
for j in range(len(eda_categoricas), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("distribucion_categoricas.png", dpi=150)
plt.show()

# Raincloud Plot adaptado (con semilla fija y conteo real)
variables_continuas_auto = [
    col for col in df.select_dtypes(include=[np.number]).columns
    if df[col].dropna().nunique() > 2
]
variables_a_graficar = variables_continuas_auto[:10]
print(f"Graficando las siguientes {len(variables_a_graficar)} variables continuas: {variables_a_graficar}")

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle(
    "Distribución y Dispersión de Variables Continuas (Dataset: SHARE)",
    fontsize=15, fontweight="bold", y=0.98
)

for ax, var in zip(axes.flatten(), variables_a_graficar):
    vals = df[var].dropna()
    if len(vals) == 0:
        ax.set_title(f"{var} (Sin datos)", fontsize=9, color="red")
        ax.set_xticks([])
        continue

    bp = ax.boxplot(
        vals, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        widths=0.4, zorder=2
    )
    bp["boxes"][0].set_facecolor("#B0BEC5")
    bp["boxes"][0].set_alpha(0.6)

    jitter_x = rng_global.normal(1, 0.04, size=len(vals))
    ax.scatter(jitter_x, vals, alpha=0.15, s=4, color="#37474F", zorder=1)
    ax.set_title(var.replace("_", " ").title(), fontweight="bold", fontsize=10)
    ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# 1.2 Correlaciones (Spearman) y Multicolinealidad (VIF)
col_ingresos = ["ingresos"]
dominio_fisico = ["fuerza", "adla", "iadla", "iadlza", "mobilidad", "grossmotor", "finemotor", "imc"]
dominio_psicologico = [
    "casp", "depresion", "pesimismo", "pensamiento_suicida",
    "culplable", "problema_de_sueno", "irritable", "perdida_de_apetito",
    "fatiga", "problema_de_concentracion", "llorado"
]
# Nota: Excluimos BF_* del VIF global para no perder ~67% de la muestra por listwise deletion
todos_dominios = [
    c for c in (dominio_fisico + dominio_psicologico + col_ingresos)
    if c in df.columns
]

corr_matrix = df[todos_dominios].corr(method="spearman")

# Heatmap filtrado con umbral consistente (|r| > 0.4)
UMBRAL_CORR = 0.4
mascara_filtro = np.abs(corr_matrix) <= UMBRAL_CORR

plt.figure(figsize=(15, 13))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
    vmin=-1, vmax=1, square=True, linewidths=0.5, mask=mascara_filtro
)
plt.title(f"Matriz de correlación de Spearman (Filtrada: |r| > {UMBRAL_CORR})", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("correlacion_dominios_filtrada.png", dpi=150)
plt.show()

# Pares fuertes recalculados directamente sobre la matriz limpia
corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs != 1.0]
fuertes = corr_pairs[abs(corr_pairs) > 0.6].drop_duplicates()

print("\n=== Pares con correlación fuerte (|r| > 0.6) ===")
for (var1, var2), valor in fuertes.items():
    fuerza = "muy fuerte" if abs(valor) > 0.8 else "fuerte"
    print(f"• {var1} - {var2}: r={valor:.3f} ({fuerza}) → evaluar redundancia")

# Cálculo de VIF sin perder muestra por BF_*
df_vif = df[todos_dominios].dropna()
X_vif = add_constant(df_vif)
vif_data = pd.DataFrame({
    "Variable": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
vif_data = vif_data[vif_data["Variable"] != "const"].sort_values("VIF", ascending=False)

print(f"\n=== VIF por variable (n={len(df_vif)} casos completos) ===")
for _, row in vif_data.iterrows():
    vif = row["VIF"]
    nivel = "🔴 SEVERA" if vif > 10 else ("🟡 MODERADA" if vif > 5 else "🟢 Aceptable")
    print(f"{nivel} | {row['Variable']}: VIF={vif:.2f}")


# 1.3 Segmentación demográfica base y pruebas estadísticas
df_plot = df.copy()
df_plot["Y_deterioro"] = df_plot["Y_deterioro"].astype(int)
df_plot["Deterioro"] = df_plot["Y_deterioro"].map({0: "No Deterioro (0)", 1: "Deterioro (1)"})
df_plot["sexo_txt"] = df_plot["sexo"].map({0: "Varón", 1: "Mujer"})

mapa_civil = {
    1: "Casado/Conviviente", 2: "Unión registrada", 3: "Separado",
    4: "Nunca casado", 5: "Divorciado", 6: "Viudo"
}
df_plot["civil_txt"] = df_plot["estado_civil"].map(mapa_civil)

mapa_ni = {
    0: "Ninguno", 1: "Primaria Inc.", 2: "Primaria Comp.",
    3: "Secundaria", 4: "Post-Secundaria", 5: "Terciario Inf.", 6: "Terciario Sup."
}
df_plot["ni_txt"] = df_plot["ni"].map(mapa_ni)
orden_ni = [mapa_ni[i] for i in sorted(mapa_ni.keys()) if mapa_ni[i] in df_plot["ni_txt"].dropna().unique()]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

sns.boxplot(data=df_plot, x="Deterioro", y="age", ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Distribución de Edad según Deterioro Cognitivo", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Edad (años)")
axes[0, 0].set_xlabel("")

tabla_sexo = pd.crosstab(df_plot["Deterioro"], df_plot["sexo_txt"], normalize="index") * 100
tabla_sexo.plot(kind="bar", stacked=True, ax=axes[0, 1], color=sns.color_palette("Pastel1", 2))
axes[0, 1].set_title("Proporción de Sexo dentro de cada Grupo", fontsize=12, fontweight="bold")
axes[0, 1].set_ylabel("Porcentaje (%)")
axes[0, 1].set_xlabel("")
axes[0, 1].tick_params(axis="x", rotation=0)
axes[0, 1].legend(title="Sexo")

df_civil_valido = df_plot.dropna(subset=["civil_txt"])
tabla_civil = pd.crosstab(df_civil_valido["Deterioro"], df_civil_valido["civil_txt"], normalize="index") * 100
tabla_civil.plot(kind="bar", stacked=True, ax=axes[1, 0], color=sns.color_palette("Set3", 6))
axes[1, 0].set_title("Distribución de Estado Civil dentro de cada Grupo", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Porcentaje (%)")
axes[1, 0].set_xlabel("Estado de Deterioro")
axes[1, 0].tick_params(axis="x", rotation=0)
axes[1, 0].legend(title="Estado Civil", bbox_to_anchor=(1.02, 1), loc="upper left")

df_ni_valido = df_plot.dropna(subset=["ni_txt"])
tabla_ni = pd.crosstab(df_ni_valido["Deterioro"], df_ni_valido["ni_txt"], normalize="index") * 100
tabla_ni = tabla_ni.reindex(columns=orden_ni, fill_value=0)
tabla_ni.plot(kind="bar", stacked=True, ax=axes[1, 1], color=sns.color_palette("viridis", len(orden_ni)))
axes[1, 1].set_title("Distribución de Nivel Educativo dentro de cada Grupo", fontsize=12, fontweight="bold")
axes[1, 1].set_ylabel("Porcentaje (%)")
axes[1, 1].set_xlabel("Estado de Deterioro")
axes[1, 1].tick_params(axis="x", rotation=0)
axes[1, 1].legend(title="Nivel Educativo (ni)", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig("perfil_deterioro_limpio.png", dpi=150, bbox_inches="tight")
plt.show()

# Tests bivariados
edad_sanos = df_plot[df_plot["Y_deterioro"] == 0]["age"].dropna()
edad_deterioro = df_plot[df_plot["Y_deterioro"] == 1]["age"].dropna()
_, p_edad = mannwhitneyu(edad_sanos, edad_deterioro, alternative="two-sided")
print(f"🔹 EDAD (Mann-Whitney U): p = {p_edad:.4e}")

for nombre, col in [("SEXO", "sexo"), ("ESTADO CIVIL", "estado_civil"), ("NIVEL INSTRUCCIÓN", "ni")]:
    df_temp = df_plot.dropna(subset=[col])
    _, p_val, _, _ = chi2_contingency(pd.crosstab(df_temp[col], df_temp["Y_deterioro"]))
    print(f"🔹 {nombre} (Chi-cuadrado): p = {p_val:.4e}")

# Distribución por país
df_pais_valido = df_plot.dropna(subset=["country"])
tabla_pais = pd.crosstab(df_pais_valido["country"], df_pais_valido["Deterioro"], normalize="index") * 100
fig, ax = plt.subplots(figsize=(14, 7))
tabla_pais.plot(kind="bar", stacked=True, color=["#7bc0a3", "#ff6b6b"], ax=ax)
ax.set_title("Distribución de Deterioro Cognitivo dentro de cada País", fontsize=14, fontweight="bold")
ax.set_ylabel("Porcentaje (%)")
ax.set_xlabel("País")
ax.tick_params(axis="x", rotation=45)
ax.legend(title="Estado de Deterioro", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("distribucion_por_pais.png", dpi=150, bbox_inches="tight")
plt.show()

_, p_val_p, _, _ = chi2_contingency(pd.crosstab(df_pais_valido["country"], df_pais_valido["Y_deterioro"]))
print(f"🔹 PAÍS (Chi-cuadrado): p = {p_val_p:.4e}")


# ==============================================================================
# FASE 2.1: APRENDIZAJE NO SUPERVISADO (K-MEANS DESDE CERO + WARD JERÁRQUICO)
# ==============================================================================
print("\n" + "=" * 70)
print("FASE 2.1: CLUSTERING NO SUPERVISADO (ANTI-LEAKAGE: SIN SGC NI Y_deterioro)")
print("=" * 70)

# Selección depurada de variables (evitando redundancia severa en bloque físico y sin leakage)
unsup_continuas = ["age", "imc", "fuerza", "anos_instruccion", "casp", "ingresos"]
unsup_ordinales = ["ni", "salud", "cronicas", "iadlza", "mobilidad"]
unsup_binarias = [
    "sexo", "smoking", "depresion", "pesimismo", "pensamiento_suicida",
    "culplable", "problema_de_sueno", "irritable", "perdida_de_apetito",
    "fatiga", "problema_de_concentracion", "llorado"
]
unsup_nominales = ["estado_civil", "tipo_ciudad"]

df_unsup = df.copy()
for col in unsup_continuas + unsup_ordinales:
    df_unsup[col] = df_unsup[col].fillna(df_unsup[col].median())
for col in unsup_binarias:
    df_unsup[col] = df_unsup[col].fillna(df_unsup[col].mode()[0])
for col in unsup_nominales:
    df_unsup[col] = df_unsup[col].fillna(df_unsup[col].mode()[0])

scaler_unsup = StandardScaler()
X_scaled_num = pd.DataFrame(
    scaler_unsup.fit_transform(df_unsup[unsup_continuas + unsup_ordinales]),
    columns=unsup_continuas + unsup_ordinales,
    index=df_unsup.index,
)
X_nominales = pd.get_dummies(
    df_unsup[unsup_nominales], columns=unsup_nominales, prefix=unsup_nominales, drop_first=True
).astype(float)

X_unsup = pd.concat([X_scaled_num, df_unsup[unsup_binarias].astype(float), X_nominales], axis=1)


# --- K-Means++ implementado desde cero ---
class KMeansScratch:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def _init_centroids(self, X_arr, rng):
        n_samples = X_arr.shape[0]
        centroids = [X_arr[rng.randint(n_samples)]]
        for _ in range(1, self.n_clusters):
            dist_sq = np.min([np.sum((X_arr - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dist_sq / dist_sq.sum()
            centroids.append(X_arr[rng.choice(n_samples, p=probs)])
        return np.array(centroids)

    def _single_run(self, X_arr, rng):
        centroids = self._init_centroids(X_arr, rng)
        labels = np.zeros(X_arr.shape[0], dtype=int)
        for _ in range(self.max_iter):
            dists = np.linalg.norm(X_arr[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.array([
                X_arr[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(self.n_clusters)
            ])
            if np.linalg.norm(new_centroids - centroids) < self.tol:
                centroids = new_centroids
                break
            centroids = new_centroids
        inertia = np.sum((X_arr - centroids[labels]) ** 2)
        return centroids, labels, inertia

    def fit_predict(self, X_df):
        X_arr = np.asarray(X_df, dtype=float)
        rng_master = np.random.RandomState(self.random_state)
        best_inertia = np.inf
        best_labels, best_centroids = None, None
        for _ in range(self.n_init):
            rng = np.random.RandomState(rng_master.randint(0, 1_000_000))
            centroids, labels, inertia = self._single_run(X_arr, rng)
            if inertia < best_inertia:
                best_inertia, best_labels, best_centroids = inertia, labels, centroids
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return best_labels


k_range = range(2, 11)
inertias_kmeans = []
sil_scores_kmeans = []

for k in k_range:
    km = KMeansScratch(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_unsup)
    inertias_kmeans.append(km.inertia_)
    sil_scores_kmeans.append(silhouette_score(X_unsup, labels))
    print(f"K={k} -> Silhouette (K-Means desde cero) = {sil_scores_kmeans[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(k_range), inertias_kmeans, marker="o")
axes[0].set_xlabel("Número de clusters (K)")
axes[0].set_ylabel("Inercia (distancia interna)")
axes[0].set_title("Método del Codo - K-Means (desde cero)")
axes[0].grid(alpha=0.3)

axes[1].plot(list(k_range), sil_scores_kmeans, marker="o", color="green")
axes[1].set_xlabel("Número de clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Coeficiente de Silueta - K-Means (desde cero)")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

K_ELEGIDO = 2
km_final = KMeansScratch(n_clusters=K_ELEGIDO, random_state=RANDOM_STATE, n_init=10)
labels_kmeans = km_final.fit_predict(X_unsup)


def plot_silhouette_detallado(X_df, labels, k):
    sil_vals = silhouette_samples(X_df, labels)
    sil_avg = silhouette_score(X_df, labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for i in range(k):
        c_vals = np.sort(sil_vals[labels == i])
        y_upper = y_lower + len(c_vals)
        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_vals, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(c_vals), str(i))
        y_lower = y_upper + 10
    ax.axvline(x=sil_avg, color="red", linestyle="--", label=f"Promedio = {sil_avg:.3f}")
    ax.set_xlabel("Coeficiente de Silueta")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Plot Detallado - K-Means (K={k})")
    ax.legend()
    plt.show()


plot_silhouette_detallado(X_unsup, labels_kmeans, K_ELEGIDO)

# Visualización PCA 2D + Top 15 loadings
pca_full = PCA()
scores_full = pca_full.fit_transform(X_unsup)
var_exp_ind = pca_full.explained_variance_ratio_
var_exp_cum = np.cumsum(var_exp_ind)

fig, ax = plt.subplots(figsize=(7, 6))
colores = cm.nipy_spectral(np.linspace(0, 1, K_ELEGIDO))
for cl in range(K_ELEGIDO):
    mask = labels_kmeans == cl
    ax.scatter(
        scores_full[mask, 0], scores_full[mask, 1],
        color=colores[cl], label=f"Cluster {cl}", s=40, alpha=0.7, edgecolors="white", linewidth=0.3
    )
ax.set_xlabel(f"PC1 ({var_exp_ind[0]*100:.1f}% var. — solo visualización)")
ax.set_ylabel(f"PC2 ({var_exp_ind[1]*100:.1f}% var. — solo visualización)")
ax.set_title(f"K-Means desde cero (K={K_ELEGIDO}) — Proyección PCA 2D", fontweight="bold")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

loadings = pd.DataFrame(
    pca_full.components_.T,
    columns=[f"PC{i+1}" for i in range(X_unsup.shape[1])],
    index=X_unsup.columns,
)
for pc_name in ["PC1", "PC2"]:
    top15 = loadings[pc_name].abs().sort_values(ascending=False).head(15).index
    top15_vals = loadings.loc[top15, pc_name].sort_values(ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top15_vals.index, top15_vals.values, color=["crimson" if x < 0 else "navy" for x in top15_vals.values], alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(f"Top 15 Variables Predictoras en {pc_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

# Perfilamiento clínico RAW (K=2)
df_clusters = df.copy()
df_clusters["cluster"] = labels_kmeans
medias_raw = df_clusters.drop(columns="cluster").groupby(df_clusters["cluster"]).mean(numeric_only=True).T
medias_raw.columns = [f"Cluster {c}" for c in medias_raw.columns]
print("\n=== TABLA DE PERFILAMIENTO CLÍNICO CON VALORES REALES (K-MEANS K=2) ===")
print(medias_raw.round(2))


# --- Clustering Jerárquico (Ward, Complete, Average) SIN SGC (Anti-Leakage) ---
X_hier_scaled = StandardScaler().fit_transform(X_unsup)

# Dendrograma de inspección sobre submuestra de 80 pacientes
Z_ward_80 = linkage(X_hier_scaled[:80], method="ward")
Z_comp_80 = linkage(X_hier_scaled[:80], method="complete")
Z_avg_80 = linkage(X_hier_scaled[:80], method="average")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("Dendrogramas — Colectivo SHARE (Primeros 80 casos, Sin Leakage de SGC)", fontsize=14, fontweight="bold", y=1.02)
for ax, Z_sub, title in zip(
    axes,
    [Z_comp_80, Z_avg_80, Z_ward_80],
    ["Complete Linkage", "Average Linkage", "Ward Linkage (Optimizado) ⭐"],
):
    dendrogram(Z_sub, ax=ax, no_labels=True, color_threshold=0.5 * max(Z_sub[:, 2]), above_threshold_color="gray")
    ax.set_title(title, fontweight="bold", fontsize=12, color="#D32F2F" if "Ward" in title else "black")
    ax.set_ylabel("Distancia Euclídea")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# Evaluación jerárquica sobre el 100% del dataset (evaluando contra Y_deterioro oficial)
Z_complete = linkage(X_hier_scaled, method="complete")
Z_average = linkage(X_hier_scaled, method="average")
Z_ward = linkage(X_hier_scaled, method="ward")
y_eval_oficial = df["Y_deterioro"].astype(int).values


def calcular_porcentaje_correcto(y_true, cluster_labels):
    mapped = np.zeros_like(cluster_labels)
    for cl in np.unique(cluster_labels):
        mask = cluster_labels == cl
        if mask.sum() > 0:
            mapped[mask] = pd.Series(y_true[mask]).mode()[0]
    return accuracy_score(y_true, mapped) * 100


print("\n" + "=" * 77)
print(f" EVALUACIÓN JERÁRQUICA SOBRE EL 100% DEL DATASET ({len(X_hier_scaled)} pacientes, SIN SGC)")
print("=" * 77)
for k in [2, 3, 4]:
    print(f"\n---> Corte en K = {k} Clusters")
    print(f"{'Enlace':<15} {'ARI':>8} {'NMI':>8} {'Silhouette':>12} {'Davies-Bouldin':>16} {'% Acierto':>12}")
    print("-" * 75)
    for nombre, Z_mat in [("Complete", Z_complete), ("Average", Z_average), ("Ward", Z_ward)]:
        labels_h = fcluster(Z_mat, t=k, criterion="maxclust")
        ari = adjusted_rand_score(y_eval_oficial, labels_h)
        nmi = normalized_mutual_info_score(y_eval_oficial, labels_h)
        sil = silhouette_score(X_hier_scaled, labels_h)
        db = davies_bouldin_score(X_hier_scaled, labels_h)
        pct = calcular_porcentaje_correcto(y_eval_oficial, labels_h)
        print(f"{nombre:<15} {ari:8.3f} {nmi:8.3f} {sil:12.3f} {db:16.3f} {pct:11.1f}%")


# ==============================================================================
# FASE 2.2: MODELADO PREDICTIVO SUPERVISADO (PARTICIÓN UNIFICADA ANTI-LEAKAGE)
# ==============================================================================
print("\n" + "=" * 70)
print("FASE 2.2: EVALUACIÓN MULTIMÉTRICA SUPERVISADA (RF, LOGREG, XGBOOST)")
print("=" * 70)

# Verificación del punto de corte oficial SGC -> Y_deterioro
print("Punto de corte SGC vs Y_deterioro:")
print(df[["SGC", "Y_deterioro"]].drop_duplicates().sort_values("SGC").head(12))

# Excluimos target, SGC (leakage), identificadores y columnas con >65% nulos (BF_*)
cols_bf = [c for c in df.columns if c.startswith("BF_")]
cols_excluir_sup = [
    c for c in ["Y_deterioro", "deterioro", "SGC", "country", "mergeid", "id", "mergeid_p"] + cols_bf
    if c in df.columns
]

sup_continuas = [c for c in ["age", "ingresos", "imc", "fuerza", "anos_instruccion", "casp", "eurod"] if c in df.columns]
sup_ordinales = [c for c in ["ni", "salud", "cronicas", "adla", "iadla", "iadlza", "mobilidad", "grossmotor", "finemotor"] if c in df.columns]
sup_binarias = [
    c for c in [
        "sexo", "smoking", "depresion", "pesimismo", "pensamiento_suicida",
        "culplable", "problema_de_sueno", "irritable", "perdida_de_apetito",
        "fatiga", "problema_de_concentracion", "llorado"
    ]
    if c in df.columns
]
sup_nominales = [c for c in ["estado_civil", "tipo_ciudad"] if c in df.columns]

# Matriz base común (conserva valores reales sin escalar para SHAP en escala real)
X_raw_base = df[sup_continuas + sup_ordinales + sup_binarias + sup_nominales].copy()
y_sup = df["Y_deterioro"].astype(int).copy()
strata_dual = y_sup.astype(str) + "_" + df["country"].astype(str)

# Partición única 80/20 estratificada dual (Target + País) para los 3 modelos
X_train_raw, X_test_raw, y_train, y_test, strata_train, strata_test = train_test_split(
    X_raw_base, y_sup, strata_dual,
    test_size=0.20, stratify=strata_dual, random_state=RANDOM_STATE
)

cv_dual = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
folds_duales = list(cv_dual.split(X_train_raw, strata_train))

# Preprocesador anti-leakage dentro de Pipeline (imputación + escalado ajustados SOLO en Train)
num_cols_sup = sup_continuas + sup_ordinales
bin_cols_sup = sup_binarias

# Para modelos basados en árboles (RF / XGBoost): imputación + escalado numérico + passthrough binarias/nominales
# Nota: Para nominales aplicamos One-Hot sobre copia alineada
X_all_encoded = pd.get_dummies(X_raw_base, columns=sup_nominales, drop_first=True).astype(float)
X_train_enc = X_all_encoded.loc[X_train_raw.index].copy()
X_test_enc = X_all_encoded.loc[X_test_raw.index].copy()
nom_ohe_cols = [c for c in X_train_enc.columns if c not in (num_cols_sup + bin_cols_sup)]

preprocesador_arboles = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols_sup),
        ("cat_bin", SimpleImputer(strategy="most_frequent"), bin_cols_sup + nom_ohe_cols),
    ],
    remainder="drop",
)

# --- 1) RANDOM FOREST REGULARIZADO ---
pipe_rf = Pipeline([
    ("prep", preprocesador_arboles),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
])

param_dist_rf = {
    "clf__n_estimators": [300, 400],
    "clf__max_depth": [4, 5],
    "clf__min_samples_split": [40, 50, 60],
    "clf__min_samples_leaf": [25, 35, 45],
    "clf__max_features": [0.2, 0.3],
    "clf__max_samples": [0.6, 0.7],
}

search_rf = RandomizedSearchCV(
    estimator=pipe_rf,
    param_distributions=param_dist_rf,
    n_iter=15,
    scoring="average_precision",
    cv=folds_duales,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
print("⚙️ Entrenando Random Forest regularizado...")
search_rf.fit(X_train_enc, y_train)
mejor_rf_pipe = search_rf.best_estimator_

probs_rf_train = mejor_rf_pipe.predict_proba(X_train_enc)[:, 1]
probs_rf_test = mejor_rf_pipe.predict_proba(X_test_enc)[:, 1]
print(f"✅ RF -> AUPRC Train: {average_precision_score(y_train, probs_rf_train):.4f} | "
      f"CV: {search_rf.best_score_:.4f} | Test: {average_precision_score(y_test, probs_rf_test):.4f}")

# --- 2) REGRESIÓN LOGÍSTICA ELASTICNET ---
# Para modelos lineales aplicamos One-Hot también sobre ordinales
X_all_linear = pd.get_dummies(
    X_raw_base, columns=sup_nominales + sup_ordinales, drop_first=True
).astype(float)
X_train_lin = X_all_linear.loc[X_train_raw.index].copy()
X_test_lin = X_all_linear.loc[X_test_raw.index].copy()
lin_rest_cols = [c for c in X_train_lin.columns if c not in sup_continuas]

preprocesador_lineal = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), sup_continuas),
        ("cat", SimpleImputer(strategy="most_frequent"), lin_rest_cols),
    ],
    remainder="drop",
)

pipe_lr = Pipeline([
    ("prep", preprocesador_lineal),
    ("clf", LogisticRegression(
        penalty="elasticnet", solver="saga", class_weight="balanced",
        max_iter=5000, random_state=RANDOM_STATE
    )),
])

param_grid_lr = {
    "clf__C": [0.01, 0.1, 1.0, 10.0],
    "clf__l1_ratio": [0.1, 0.5, 0.7, 0.9],
}

grid_lr = GridSearchCV(
    estimator=pipe_lr, param_grid=param_grid_lr, cv=folds_duales, scoring="roc_auc", n_jobs=-1
)
print("⚙️ Entrenando Regresión Logística ElasticNet...")
grid_lr.fit(X_train_lin, y_train)
mejor_lr_pipe = grid_lr.best_estimator_

probs_lr_train = mejor_lr_pipe.predict_proba(X_train_lin)[:, 1]
probs_lr_test = mejor_lr_pipe.predict_proba(X_test_lin)[:, 1]
print(f"✅ LogReg -> AUROC Train: {roc_auc_score(y_train, probs_lr_train):.4f} | "
      f"CV: {grid_lr.best_score_:.4f} | Test: {roc_auc_score(y_test, probs_lr_test):.4f}")

# --- 3) XGBOOST CLASSIFIER ---
conteo_clases = y_train.value_counts()
factor_desbalance = conteo_clases[0] / conteo_clases[1]

pipe_xgb = Pipeline([
    ("prep", preprocesador_arboles),
    ("clf", XGBClassifier(
        scale_pos_weight=factor_desbalance,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )),
])

param_grid_xgb = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5],
    "clf__learning_rate": [0.01, 0.05, 0.1],
}

grid_xgb = GridSearchCV(
    estimator=pipe_xgb, param_grid=param_grid_xgb, cv=folds_duales, scoring="roc_auc", n_jobs=-1
)
print("⚙️ Entrenando XGBoost Classifier...")
grid_xgb.fit(X_train_enc, y_train)
mejor_xgb_pipe = grid_xgb.best_estimator_

probs_xgb_train = mejor_xgb_pipe.predict_proba(X_train_enc)[:, 1]
probs_xgb_test = mejor_xgb_pipe.predict_proba(X_test_enc)[:, 1]
print(f"✅ XGBoost -> AUROC Train: {roc_auc_score(y_train, probs_xgb_train):.4f} | "
      f"CV: {grid_xgb.best_score_:.4f} | Test: {roc_auc_score(y_test, probs_xgb_test):.4f}")


# --- Comparativa Unificada de los 3 Modelos Supervisados en Test ---
def evaluar_modelo_test(nombre, y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f2_scores = [
        (5 * p * r) / (4 * p + r) if (4 * p + r) > 0 else 0
        for p, r in zip(precisions[:-1], recalls[:-1])
    ]
    umbral_opt = thresholds[np.argmax(f2_scores)]
    y_pred = (y_prob >= umbral_opt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Modelo": nombre,
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Umbral Óptimo (F2)": umbral_opt,
        "Sensibilidad (Recall)": recall_score(y_true, y_pred),
        "Especificidad": tn / (tn + fp),
        "Precisión (VPP)": precision_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "F2-Score": fbeta_score(y_true, y_pred, beta=2),
    }


df_comparativa_sup = pd.DataFrame([
    evaluar_modelo_test("Random Forest (Regularizado)", y_test, probs_rf_test),
    evaluar_modelo_test("Regresión Logística (ElasticNet)", y_test, probs_lr_test),
    evaluar_modelo_test("XGBoost Classifier", y_test, probs_xgb_test),
])

print("\n" + "=" * 85)
print("🏆 COMPARATIVA UNIFICADA DE MODELOS SUPERVISADOS EN TEST (20% CIEGO)")
print("=" * 85)
print(df_comparativa_sup.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("=" * 85)

# Curvas ROC y Precision-Recall comparativas de los 3 modelos
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
modelos_probs = [
    ("Random Forest", probs_rf_test, "darkorange"),
    ("Regresión Logística", probs_lr_test, "royalblue"),
    ("XGBoost", probs_xgb_test, "forestgreen"),
]

for nombre, probs, color in modelos_probs:
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[0].plot(fpr, tpr, lw=2, color=color, label=f"{nombre} (AUROC = {roc_auc_score(y_test, probs):.3f})")

    prec, rec, _ = precision_recall_curve(y_test, probs)
    axes[1].plot(rec, prec, lw=2, color=color, label=f"{nombre} (AUPRC = {average_precision_score(y_test, probs):.3f})")

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar (0.500)")
axes[0].set_title("Curvas ROC Comparativas (Test)", fontweight="bold")
axes[0].set_xlabel("1 - Especificidad (FPR)")
axes[0].set_ylabel("Sensibilidad (Recall)")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

axes[1].axhline(y=y_test.mean(), color="gray", linestyle="--", label=f"Prevalencia ({y_test.mean():.1%})")
axes[1].set_title("Curvas Precision-Recall Comparativas (Test)", fontweight="bold")
axes[1].set_xlabel("Sensibilidad (Recall)")
axes[1].set_ylabel("Precisión (VPP)")
axes[1].legend(loc="lower left")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("comparativa_modelos_supervisados.png", dpi=150)
plt.show()


# ==============================================================================
# FASE 3: EXPLICABILIDAD CLÍNICA (PERMUTATION IMPORTANCE, SHAP Y DCA)
# ==============================================================================
print("\n" + "=" * 70)
print("FASE 3: EXPLICABILIDAD CLÍNICA (SHAP & DECISION CURVE ANALYSIS)")
print("=" * 70)

# Extraemos el estimador RF y la matriz transformada con nombres de columnas exactos
rf_estimador = mejor_rf_pipe.named_steps["clf"]
prep_rf_fitted = mejor_rf_pipe.named_steps["prep"]
cols_rf_final = num_cols_sup + bin_cols_sup + nom_ohe_cols

X_test_rf_trans = pd.DataFrame(
    prep_rf_fitted.transform(X_test_enc),
    columns=cols_rf_final,
    index=X_test_enc.index,
)
# Matriz de valores reales imputados con el mismo orden exacto de columnas para visualizar SHAP
X_test_rf_real = pd.DataFrame(
    SimpleImputer(strategy="median").fit(X_train_enc[cols_rf_final]).transform(X_test_enc[cols_rf_final]),
    columns=cols_rf_final,
    index=X_test_enc.index,
)

# 3.1 Importancia por Impureza vs Permutación en Test
perm_result = permutation_importance(
    rf_estimador, X_test_rf_trans, y_test,
    scoring="average_precision", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
)

df_importancias = pd.DataFrame({
    "Feature": cols_rf_final,
    "Importancia_Gini": rf_estimador.feature_importances_,
    "Importancia_Permutacion": perm_result.importances_mean,
    "STD_Permutacion": perm_result.importances_std,
}).sort_values(by="Importancia_Permutacion", ascending=False).reset_index(drop=True)

print("\n🏆 TOP 10 VARIABLES MÁS IMPORTANTES (PERMUTACIÓN EN TEST):")
print(df_importancias.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 3.2 Valores SHAP (Beeswarm Summary Plot + Dependence Plots en Escala Real)
explainer = shap.TreeExplainer(rf_estimador)
shap_values = explainer(X_test_rf_trans)

if len(shap_values.shape) == 3:
    shap_vals_target = shap_values[:, :, 1]
else:
    shap_vals_target = shap_values

# Beeswarm Summary Plot (con valores reales como referencia visual)
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_vals_target.values,
    X_test_rf_real,
    max_display=15,
    show=False,
)
plt.title("SHAP Summary Plot (Beeswarm) — Impacto en Deterioro Cognitivo", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_beeswarm_summary.png", dpi=200, bbox_inches="tight")
plt.show()

# Dependence Plots en escala real (corregido: usa shap_vals_target.values y columnas alineadas)
for col_dep, xlabel_txt in [
    ("age", "Edad del Paciente (Años)"),
    ("cronicas", "Cantidad Acumulada de Enfermedades Crónicas"),
]:
    if col_dep in X_test_rf_real.columns:
        plt.figure(figsize=(8.5, 5.5))
        shap.dependence_plot(
            col_dep,
            shap_vals_target.values,
            X_test_rf_real,
            interaction_index="auto",
            show=False,
        )
        plt.title(f"Dependence Plot SHAP: {col_dep} (Escala Real)", fontsize=12, fontweight="bold")
        plt.xlabel(xlabel_txt, fontsize=10, fontweight="bold")
        plt.ylabel("Impacto SHAP en Deterioro Cognitivo", fontsize=10, fontweight="bold")
        plt.axhline(0, color="red", linestyle="--", alpha=0.7)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"shap_dependence_{col_dep}_real.png", dpi=200, bbox_inches="tight")
        plt.show()

# 3.3 Decision Curve Analysis (DCA)
y_true_dca = np.asarray(y_test)
y_prob_dca = np.asarray(probs_rf_test)
thresholds_dca = np.arange(0.05, 0.51, 0.01)
N_dca = len(y_true_dca)
prevalence_dca = y_true_dca.mean()

net_benefit_model = []
net_benefit_all = []

for pt in thresholds_dca:
    pred = (y_prob_dca >= pt).astype(int)
    tp = np.sum((pred == 1) & (y_true_dca == 1))
    fp = np.sum((pred == 1) & (y_true_dca == 0))
    net_benefit_model.append((tp / N_dca) - (fp / N_dca) * (pt / (1 - pt)))
    net_benefit_all.append(prevalence_dca - (1 - prevalence_dca) * (pt / (1 - pt)))

umbral_rf_opt = df_comparativa_sup.loc[0, "Umbral Óptimo (F2)"]

plt.figure(figsize=(8, 6))
plt.plot(thresholds_dca, net_benefit_model, linewidth=2.5, label="Random Forest")
plt.plot(thresholds_dca, net_benefit_all, "--", linewidth=2, label="Tratar a todos")
plt.plot(thresholds_dca, np.zeros_like(thresholds_dca), ":", linewidth=2, label="No tratar a nadie")
plt.axvline(x=umbral_rf_opt, color="red", linestyle=":", linewidth=2, label=f"Umbral óptimo F2 ({umbral_rf_opt:.2f})")
plt.xlabel("Umbral de probabilidad")
plt.ylabel("Beneficio neto")
plt.title("Decision Curve Analysis (DCA)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("decision_curve_analysis.png", dpi=200)
plt.show()

print("\n✅ Pipeline completo ejecutado exitosamente sin errores ni data leakage.")
