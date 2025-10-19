# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# 👇 Importar la función del recomendador
from .recomendador_xgboost import generar_recomendacion


# Crear la instancia de FastAPI
app = FastAPI(title="XGBoost Prediction API")

# Habilitar CORS
origins = ["http://localhost:4200"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "xgboost_model_final.pkl")
model = joblib.load(model_path)

# Definir los campos esperados
class InputData(BaseModel):
    activo_corriente: int
    capital_invertido: int
    departamento: int
    provincia: int
    gastos_financieros: int
    gastos_por_prestamos: float
    deudas_corto_plazo: int
    patrimonio_empresa: float
    ganancia_operativa: int
    total_activo: float
    deudas_largo_plazo: float
    total_pasivo_y_patrimonio: float
    numero_empleados: int
    ventas_netas: int
    canal_principal: int
    cantidad_canales_venta: int
    vende_por_catalogo: bool
    vende_con_comisionistas: bool
    vende_a_domicilio: bool
    vende_en_linea: bool
    vende_en_ferias: bool
    vende_en_maquinas_venta: bool
    vende_en_tienda: bool
    otros_medios_venta: bool
    vende_por_telefono: bool
    uso_de_software: bool


@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}


# 🧠 Nuevo endpoint para recomendaciones
@app.post("/recomendar")
def recomendar(data: InputData):
    df = pd.DataFrame([data.dict()])

    # 🔹 Calcular probabilidad de supervivencia
    prob = float(model.predict_proba(df)[0][1])

    # 🔹 Generar recomendaciones con contexto
    recomendaciones = generar_recomendacion(df, idx=0, probabilidad=prob)

    return {
        "probabilidad_sobrevivir": round(prob, 4),
        "recomendaciones": recomendaciones
    }

