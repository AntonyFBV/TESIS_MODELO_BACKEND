# api/recomendador_xgboost.py
# -*- coding: utf-8 -*-
import shap
import pandas as pd
import xgboost as xgb
import os
import numpy as np

# === 1️⃣ Cargar el modelo una sola vez ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "xgboost_model_final.json")

model = xgb.XGBClassifier()
model.load_model(model_path)

# === 2️⃣ Columnas según InputData actualizado ===
columnas = [
    "capital_invertido", "provincia", "gastos_por_prestamos", "deudas_corto_plazo",
    "patrimonio_empresa", "total_activo", "numero_empleados", "ventas_netas",
    "canal_principal", "cantidad_canales_venta", "vende_por_catalogo",
    "vende_con_comisionistas", "vende_a_domicilio", "vende_en_linea",
    "vende_en_ferias", "vende_en_maquinas_venta", "vende_en_tienda",
    "otros_medios_venta", "vende_por_telefono", "publicidad", "uso_de_software"
]

# === 3️⃣ Crear explainer SHAP una sola vez ===
X_base = pd.DataFrame([{col: 0 for col in columnas}])
explainer = shap.Explainer(model, X_base)

# === 4️⃣ Tipos de columnas ===
numericas = [
    "capital_invertido", "gastos_por_prestamos", "deudas_corto_plazo",
    "patrimonio_empresa", "total_activo", "numero_empleados", "ventas_netas"
]

categoricas = [
    "cantidad_canales_venta"
]

booleanas = [
    "vende_por_catalogo", "vende_con_comisionistas", "vende_a_domicilio",
    "vende_en_linea", "vende_en_ferias", "vende_en_maquinas_venta",
    "vende_en_tienda", "otros_medios_venta", "vende_por_telefono",
    "publicidad", "uso_de_software"
]

# === 5️⃣ Variables relevantes (todas excepto provincia y canal_principal) ===
variables_relevantes = numericas + categoricas + booleanas

# === 6️⃣ Función de recomendación completa ===
def generar_recomendacion(X: pd.DataFrame, idx: int = 0, probabilidad: float = None):
    """
    Genera recomendaciones basadas en los valores SHAP.
    - X: DataFrame con una o más filas (una empresa).
    - idx: índice de la empresa dentro de X.
    - probabilidad: probabilidad estimada de supervivencia (opcional, mejora el mensaje).
    """
    # Asegurar que todo sea numérico y sin NaN
    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes

    # Reordenar columnas para que coincidan con el explainer
    X = X.reindex(columns=columnas, fill_value=0)

    # Calcular valores SHAP
    shap_values = explainer(X)
    shap_values_instance = shap_values[idx].values
    features = X.iloc[idx]

    # Filtrar solo variables relevantes
    feature_importances = [
        (f, features[f], shap_values_instance[i])
        for i, f in enumerate(X.columns)
        if f in variables_relevantes
    ]

    # Ordenar todas las variables por impacto absoluto
    feature_importances = sorted(
        feature_importances,
        key=lambda x: abs(x[2]),
        reverse=True
    )

    # Calcular impacto total para normalización
    total_shap_abs = sum(abs(v) for _, _, v in feature_importances) or 1.0

    # Generar lista de recomendaciones
    recomendaciones = []
    for feature, value, impact in feature_importances:
        # Valor legible según tipo
        if feature in booleanas:
            valor_mostrar = "Sí" if value == 1 else "No"
        elif feature in numericas:
            valor_mostrar = float(value)
        else:
            valor_mostrar = int(value)

        # Mensaje neutro según el impacto
        if impact > 0:
            mensaje = f"Revisar '{feature}', podría estar contribuyendo positivamente a la supervivencia."
        else:
            mensaje = f"Revisar '{feature}', podría estar afectando negativamente la supervivencia."

        # Impacto relativo en %
        impacto_pct = 100 * abs(impact) / total_shap_abs

        recomendaciones.append({
            "variable": feature,
            "valor_actual": valor_mostrar,
            "impacto": round(float(impact), 4),
            "impacto_pct": round(impacto_pct, 2),
            "recomendacion": mensaje
        })

    # === Contexto general ===
    contexto = ""
    if probabilidad is not None:
        if probabilidad < 0.4:
            contexto = f"La probabilidad estimada de supervivencia es baja ({round(probabilidad*100,2)}%)."
        elif probabilidad < 0.7:
            contexto = f"La probabilidad estimada de supervivencia es moderada ({round(probabilidad*100,2)}%)."
        else:
            contexto = f"La probabilidad estimada de supervivencia es alta ({round(probabilidad*100,2)}%)."

    return {
        "contexto": contexto,
        "recomendaciones": recomendaciones
    }
