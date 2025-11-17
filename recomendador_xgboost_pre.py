# recomendador_xgboost.py
# -*- coding: utf-8 -*-
import pandas as pd
import shap
import joblib
from sqlalchemy import create_engine

# === 1️⃣ Cargar modelo entrenado ===
print("🔄 Cargando modelo entrenado...")
model = joblib.load("xgboost_model_final.pkl")
print("✅ Modelo cargado")

# === 2️⃣ Conexión a la base de datos ===
engine = create_engine("postgresql+psycopg2://postgres:Fabio123@localhost:5432/modelo_empresas")

# === 3️⃣ Cargar datos de ejemplo (solo si querés probar localmente) ===
df = pd.read_sql("SELECT * FROM public.empresa_datos", engine)
X = df.drop('sobrevive_2años', axis=1)

# --- 🔧 LIMPIEZA ROBUSTA: asegurar que todo sea numérico ---
for col in X.columns:
    # Si la columna tiene strings o tipos object
    if X[col].dtype == 'object' or str(X[col].dtype).startswith('string'):
        try:
            # Intentar convertir directamente
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            pass
    # Si quedan valores NaN o None, reemplazar por 0
    X[col] = X[col].fillna(0)

# Si aún quedan columnas con tipo object, codificarlas a números
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

# --- 🔍 Confirmar tipos finales ---
print("Tipos de datos finales de X:")
print(X.dtypes)
print("¿Hay columnas tipo object?:", any(X.dtypes == 'object'))

# === 4️⃣ Crear el explainer de SHAP ===
print("🧠 Creando explainer SHAP...")
explainer = shap.TreeExplainer(model)  # usa TreeExplainer directamente
print("✅ Explainer listo")


# === 5️⃣ Función para generar recomendaciones ===
def generar_recomendacion(X, idx=0):
    shap_values = explainer(X.iloc[[idx]])
    valores = shap_values.values[0]
    features = X.columns

    impacto = pd.DataFrame({
        'variable': features,
        'impacto': valores,
        'valor_actual': X.iloc[idx].values
    }).sort_values('impacto')

    # Variables que más empujan hacia False
    negativas = impacto.head(3)

    recomendaciones = []
    for _, row in negativas.iterrows():
        if row['impacto'] < 0:
            recomendaciones.append({
                'variable': row['variable'],
                'valor_actual': row['valor_actual'],
                'impacto': round(row['impacto'], 4),
                'recomendacion': f"Aumentar o mejorar '{row['variable']}' puede aumentar la probabilidad de supervivencia."
            })
    return recomendaciones

# === 6️⃣ Ejemplo de uso ===
if __name__ == "__main__":
    idx = 0  # índice del caso que querés analizar
    print(f"\n📊 Recomendaciones para empresa #{idx}")
    recs = generar_recomendacion(X, idx)
    for r in recs:
        print(f"🔹 {r['recomendacion']} (valor actual: {r['valor_actual']}, impacto: {r['impacto']})")
