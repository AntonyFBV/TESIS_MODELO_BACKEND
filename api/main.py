# api/main.py
from fastapi import FastAPI
import pandas as pd
import joblib
import os
from pydantic import BaseModel

# 🔹 Asegurar ruta correcta del modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "xgboost_model_final.pkl")
model = joblib.load(model_path)

app = FastAPI(title="XGBoost Prediction API")

# 🔹 Definir los campos que espera el modelo
class InputData(BaseModel):
    activo_corriente: int
    capital_invertido: int
    departamento: int
    provincia: int
    gastos_financieros: int
    gastos_por_prestamos: float
    deudas_corto_plazo: int
    patrimonio_empresa: float
    ganancia_operativa: float
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

# 🔹 Endpoint de predicción
@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}
