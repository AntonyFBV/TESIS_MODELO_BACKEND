import pandas as pd
import joblib

# Cargar modelo
model = joblib.load("model/xgboost_model_final.pkl")

# Crear inputs de prueba
tests = [
    {"activo_corriente": 0, "capital_invertido":0, "departamento":0, "provincia":0,
     "gastos_financieros":0, "gastos_por_prestamos":0.0, "deudas_corto_plazo":0,
     "patrimonio_empresa":0.0, "ganancia_operativa":0, "total_activo":0.0,
     "deudas_largo_plazo":0.0, "total_pasivo_y_patrimonio":0.0, "numero_empleados":0,
     "ventas_netas":0, "canal_principal":0, "cantidad_canales_venta":0,
     "vende_por_catalogo":False, "vende_con_comisionistas":False,
     "vende_a_domicilio":False, "vende_en_linea":False, "vende_en_ferias":False,
     "vende_en_maquinas_venta":False, "vende_en_tienda":False,
     "otros_medios_venta":False, "vende_por_telefono":False, "uso_de_software":False},
    # aquí podrías agregar valores promedio o máximos
]

df = pd.DataFrame(tests)

# Predicción y probabilidad
pred = model.predict(df)
proba = model.predict_proba(df) if hasattr(model, "predict_proba") else None

print("Predicción:", pred)
if proba is not None:
    print("Probabilidades:", proba)
