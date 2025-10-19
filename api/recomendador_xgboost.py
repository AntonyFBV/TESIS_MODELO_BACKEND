# api/recomendador_xgboost.py
# -*- coding: utf-8 -*-
import shap
import pandas as pd
import joblib
import os

# === 1️⃣ Cargar el modelo una sola vez ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "xgboost_model_final.pkl")
model = joblib.load(model_path)

# === 2️⃣ Crear explainer SHAP una sola vez ===
# (Usamos una fila base con las columnas esperadas)
columnas = [
    'activo_corriente', 'capital_invertido', 'departamento', 'provincia',
    'gastos_financieros', 'gastos_por_prestamos', 'deudas_corto_plazo',
    'patrimonio_empresa', 'ganancia_operativa', 'total_activo', 'deudas_largo_plazo',
    'total_pasivo_y_patrimonio', 'numero_empleados', 'ventas_netas',
    'canal_principal', 'cantidad_canales_venta', 'vende_por_catalogo',
    'vende_con_comisionistas', 'vende_a_domicilio', 'vende_en_linea',
    'vende_en_ferias', 'vende_en_maquinas_venta', 'vende_en_tienda',
    'otros_medios_venta', 'vende_por_telefono', 'uso_de_software'
]
X_base = pd.DataFrame([{col: 0 for col in columnas}])
explainer = shap.Explainer(model, X_base)


# === 3️⃣ Función de recomendación ===
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

    # Calcular valores SHAP
    shap_values = explainer(X)
    shap_values_instance = shap_values[idx].values
    features = X.iloc[idx]

    # Ordenar las 3 variables con mayor impacto negativo
    feature_importances = sorted(
        zip(X.columns, features.values, shap_values_instance),
        key=lambda x: x[2]
    )[:3]

    # Generar recomendaciones personalizadas
    recomendaciones = []
    for feature, value, impact in feature_importances:
        if probabilidad is not None:
            if probabilidad >= 0.7:
                mensaje = f"Mantener o reforzar '{feature}' para conservar la buena posición de la empresa."
            elif probabilidad >= 0.5:
                mensaje = f"Mejorar ligeramente '{feature}' podría fortalecer la probabilidad de supervivencia."
            else:
                mensaje = f"Aumentar o mejorar '{feature}' puede aumentar significativamente la probabilidad de supervivencia."
        else:
            mensaje = f"Aumentar o mejorar '{feature}' puede aumentar la probabilidad de supervivencia."

        recomendaciones.append({
            "variable": feature,
            "valor_actual": float(value),
            "impacto": round(float(impact), 4),
            "recomendacion": mensaje
        })

    return recomendaciones
