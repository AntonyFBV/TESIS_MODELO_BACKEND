import shap
import pandas as pd
import requests
from api.core.config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL
from api.core.model_loader import get_model
from .features import COLUMNAS, NUMERICAS, CATEGORICAS, BOOLEANAS, VARIABLES_RELEVANTES
from .preprocessing import preparar_input

# ─── SHAP EXPLAINER (inicializado una sola vez) ────────────────────────────────
_X_base = pd.DataFrame([{col: 0 for col in COLUMNAS}])
_explainer = shap.Explainer(get_model(), _X_base)

# ─── AUTORES POR DOMINIO ───────────────────────────────────────────────────────
_AUTORES = {
    "canal":     ["Kotler y Keller (2016)", "Anderson y Kumar (2006)", "Stern y El-Ansary (1996)"],
    "vende":     ["Kotler y Keller (2016)", "Anderson y Kumar (2006)", "Stern y El-Ansary (1996)"],
    "publicidad":["Kotler y Armstrong (2018)", "Aaker (1991)", "Ries y Trout (2001)"],
    "empleados": ["Chiavenato (2009)", "Dessler (2017)", "Robbins y Judge (2013)"],
    "software":  ["Brynjolfsson y McAfee (2014)", "Rogers (2003)", "Venkatraman (1994)"],
    "financiero":["Gitman y Zutter (2016)", "Ross et al. (2019)", "Brigham y Ehrhardt (2018)"],
    "default":   ["Porter (1985)", "Drucker (2006)", "Kaplan y Norton (1996)"],
}

_ALIAS = {"gastos_por_prestamos": "margen comercial (ganancias)"}


def _autores_para(feature: str) -> list[str]:
    for key in ("vende", "canal", "publicidad", "empleados", "software"):
        if key in feature:
            return _AUTORES[key]
    for key in ("capital", "activo", "patrimonio", "deuda", "margen", "ganancia"):
        if key in feature:
            return _AUTORES["financiero"]
    return _AUTORES["default"]


# ─── LLAMADA A GROQ ────────────────────────────────────────────────────────────
def _recomendacion_ia(feature: str, valor_actual, impacto: float, probabilidad: float) -> str:
    efecto = "positivamente" if impacto > 0 else "negativamente"
    nombre = _ALIAS.get(feature, feature)
    autores = _autores_para(feature)

    prompt = (
        f"Eres un consultor empresarial experto. Analiza esta métrica de una empresa:\n"
        f"Variable: {nombre}\nValor actual: {valor_actual}\n"
        f"Impacto en supervivencia: {efecto} ({round(impacto, 4)})\n"
        f"Probabilidad de supervivencia: {round(probabilidad * 100, 2)}%\n\n"
        f"Genera UNA recomendación breve (máximo 3 líneas) que incluya:\n"
        f"1. Cita de UN SOLO autor de esta lista (elige el más relevante): {', '.join(autores)}\n"
        f"2. Explicación del impacto\n"
        f"3. Recomendación específica y accionable\n\n"
        f'Formato exacto:\n"Según [Autor] ([Año]), [fundamento teórico específico]. '
        f"Esta variable está impactando {efecto} porque [razón contextual]. "
        f'Recomendación: [acción concreta y práctica]."\n\n'
        f"IMPORTANTE: Usa SOLO UN autor de la lista. Sé específico. Evita repetir palabras entre recomendaciones."
    )

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "Eres un experto en gestión empresarial que cita autores reconocidos."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 200,
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        print(f"⚠️ Groq {response.status_code}: {response.text}")
    except Exception as e:
        print(f"⚠️ Groq error: {e}")

    return f"Revisar '{feature}' ya que está impactando {efecto} la supervivencia del negocio."


# ─── FUNCIÓN PRINCIPAL ─────────────────────────────────────────────────────────
def generar_recomendacion(X: pd.DataFrame, idx: int = 0, probabilidad: float = None) -> dict:
    X_clean = preparar_input(X)
    shap_values = _explainer(X_clean)
    shap_vals = shap_values[idx].values
    features = X_clean.iloc[idx]

    importancias = sorted(
        [
            (f, features[f], shap_vals[i])
            for i, f in enumerate(X_clean.columns)
            if f in VARIABLES_RELEVANTES
        ],
        key=lambda x: abs(x[2]),
        reverse=True,
    )[:7]

    total_abs = sum(abs(v) for _, _, v in importancias) or 1.0
    prob = probabilidad or 0.5

    recomendaciones = []
    for feature, value, impact in importancias:
        if feature in BOOLEANAS:
            valor_mostrar = "Sí" if value == 1 else "No"
        elif feature in NUMERICAS:
            valor_mostrar = float(value)
        else:
            valor_mostrar = int(value)

        recomendaciones.append({
            "variable": feature,
            "valor_actual": valor_mostrar,
            "impacto": round(float(impact), 4),
            "impacto_pct": round(100 * abs(impact) / total_abs, 2),
            "recomendacion": _recomendacion_ia(feature, valor_mostrar, impact, prob),
        })

    if prob < 0.4:
        contexto = f"⚠️ La probabilidad estimada de supervivencia es baja ({round(prob*100,2)}%). Es crítico actuar sobre las variables de mayor impacto."
    elif prob < 0.7:
        contexto = f"⚡ La probabilidad estimada de supervivencia es moderada ({round(prob*100,2)}%). Hay margen de mejora significativo."
    else:
        contexto = f"✅ La probabilidad estimada de supervivencia es alta ({round(prob*100,2)}%). Mantener las buenas prácticas actuales."

    return {"contexto": contexto, "recomendaciones": recomendaciones}
