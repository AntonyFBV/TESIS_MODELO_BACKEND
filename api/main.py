# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import xgboost as xgb  # ✅ usar xgboost directamente

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

# ✅ Cargar modelo XGBoost en formato JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "xgboost_model_final.json")

model = xgb.XGBClassifier()
model.load_model(model_path)

# ✅ Fin del cambio

# Definir los campos esperados
class InputData(BaseModel):
    capital_invertido: int
    provincia: int
    gastos_por_prestamos: float
    deudas_corto_plazo: int
    patrimonio_empresa: float
    total_activo: float
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
    publicidad:bool
    uso_de_software: bool


@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}


@app.post("/recomendar")
def recomendar(data: InputData):
    df = pd.DataFrame([data.dict()])

    prob = float(model.predict_proba(df)[0][1])
    recomendaciones = generar_recomendacion(df, idx=0, probabilidad=prob)

    return {
        "probabilidad_sobrevivir": round(prob, 4),
        "recomendaciones": recomendaciones
    }
