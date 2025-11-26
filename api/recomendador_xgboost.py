# api/recomendador_xgboost.py
# -*- coding: utf-8 -*-
import shap
import pandas as pd
import xgboost as xgb
import os
import requests
import json

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "xgboost_model_final.json")

model = xgb.XGBClassifier()
model.load_model(model_path)

columnas = [
    "capital_invertido", "provincia", "gastos_por_prestamos", "deudas_corto_plazo",
    "patrimonio_empresa", "total_activo", "numero_empleados", "ventas_netas",
    "canal_principal", "cantidad_canales_venta", "vende_por_catalogo",
    "vende_con_comisionistas", "vende_a_domicilio", "vende_en_linea",
    "vende_en_ferias", "vende_en_maquinas_venta", "vende_en_tienda",
    "otros_medios_venta", "vende_por_telefono", "publicidad", "uso_de_software"
]

X_base = pd.DataFrame([{col: 0 for col in columnas}])
explainer = shap.Explainer(model, X_base)

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

variables_relevantes = numericas + categoricas + booleanas


def generar_recomendacion_ia(feature: str, valor_actual, impacto: float, probabilidad: float):
    """
    Usa Groq (Llama 3.1) para generar recomendaciones fundamentadas con autores.
    Hace la llamada HTTP directamente sin librerías adicionales.
    """
    # Determinar si es positivo o negativo
    efecto = "positivamente" if impacto > 0 else "negativamente"
    
    nombre_mostrar = feature
    if feature == "gastos_por_prestamos":
        nombre_mostrar = "margen comercial (ganancias)"
    
    autores_sugeridos = []
    if "vende" in feature or "canal" in feature:
        autores_sugeridos = ["Kotler y Keller (2016)", "Anderson y Kumar (2006)", "Stern y El-Ansary (1996)"]
    elif "publicidad" in feature or "marketing" in feature:
        autores_sugeridos = ["Kotler y Armstrong (2018)", "Aaker (1991)", "Ries y Trout (2001)"]
    elif "empleados" in feature or "personal" in feature:
        autores_sugeridos = ["Chiavenato (2009)", "Dessler (2017)", "Robbins y Judge (2013)"]
    elif "software" in feature or "tecnolog" in feature:
        autores_sugeridos = ["Brynjolfsson y McAfee (2014)", "Rogers (2003)", "Venkatraman (1994)"]
    elif "capital" in feature or "activo" in feature or "patrimonio" in feature or "deuda" in feature or "margen" in feature or "ganancia" in feature:
        autores_sugeridos = ["Gitman y Zutter (2016)", "Ross et al. (2019)", "Brigham y Ehrhardt (2018)"]
    else:
        autores_sugeridos = ["Porter (1985)", "Drucker (2006)", "Kaplan y Norton (1996)"]
    
    prompt = f"""Eres un consultor empresarial experto. Analiza esta métrica de una empresa:

Variable: {nombre_mostrar}
Valor actual: {valor_actual}
Impacto en supervivencia: {efecto} ({round(impacto, 4)})
Probabilidad de supervivencia: {round(probabilidad * 100, 2)}%

Genera UNA recomendación breve (máximo 3 líneas) que incluya:
1. Cita de UN SOLO autor de esta lista (elige el más relevante): {', '.join(autores_sugeridos)}
2. Explicación del impacto
3. Recomendación específica y accionable

Formato exacto:
"Según [Autor] ([Año]), [fundamento teórico específico]. Esta variable está impactando {efecto} porque [razón contextual]. Recomendación: [acción concreta y práctica]."

IMPORTANTE: 
- Usa SOLO UN autor de la lista sugerida
- Sé específico con el fundamento teórico (no genérico)
- La recomendación debe ser práctica y accionable
- Evita repetir las mismas palabras en diferentes recomendaciones"""

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "Eres un experto en gestión empresarial que cita autores reconocidos para fundamentar recomendaciones."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }
        
        # Hacer request a Groq API
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            print(f"⚠️ Error Groq API: {response.status_code} - {response.text}")
            return f"Revisar '{feature}' ya que está impactando {efecto} la supervivencia del negocio."
    
    except Exception as e:
        # Si falla la API, retornar mensaje genérico
        print(f"⚠️ Error al llamar Groq API: {e}")
        return f"Revisar '{feature}' ya que está impactando {efecto} la supervivencia del negocio."


def generar_recomendacion(X: pd.DataFrame, idx: int = 0, probabilidad: float = None):
    """
    Genera recomendaciones basadas en los valores SHAP usando IA para mejorar los mensajes.
    """
    X = X.fillna(0)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes

    X = X.reindex(columns=columnas, fill_value=0)

    shap_values = explainer(X)
    shap_values_instance = shap_values[idx].values
    features = X.iloc[idx]

    feature_importances = [
        (f, features[f], shap_values_instance[i])
        for i, f in enumerate(X.columns)
        if f in variables_relevantes
    ]

    feature_importances = sorted(
        feature_importances,
        key=lambda x: abs(x[2]),
        reverse=True
    )[:7]  

    total_shap_abs = sum(abs(v) for _, _, v in feature_importances) or 1.0

    recomendaciones = []
    for feature, value, impact in feature_importances:
        if feature in booleanas:
            valor_mostrar = "Sí" if value == 1 else "No"
        elif feature in numericas:
            valor_mostrar = float(value)
        else:
            valor_mostrar = int(value)

        mensaje_ia = generar_recomendacion_ia(
            feature=feature,
            valor_actual=valor_mostrar,
            impacto=impact,
            probabilidad=probabilidad or 0.5
        )

        # Impacto relativo en %
        impacto_pct = 100 * abs(impact) / total_shap_abs

        recomendaciones.append({
            "variable": feature,
            "valor_actual": valor_mostrar,
            "impacto": round(float(impact), 4),
            "impacto_pct": round(impacto_pct, 2),
            "recomendacion": mensaje_ia  
        })

    contexto = ""
    if probabilidad is not None:
        if probabilidad < 0.4:
            contexto = f"⚠️ La probabilidad estimada de supervivencia es baja ({round(probabilidad*100,2)}%). Es crítico actuar sobre las variables de mayor impacto."
        elif probabilidad < 0.7:
            contexto = f"⚡ La probabilidad estimada de supervivencia es moderada ({round(probabilidad*100,2)}%). Hay margen de mejora significativo."
        else:
            contexto = f"✅ La probabilidad estimada de supervivencia es alta ({round(probabilidad*100,2)}%). Mantener las buenas prácticas actuales."

    return {
        "contexto": contexto,
        "recomendaciones": recomendaciones
    }