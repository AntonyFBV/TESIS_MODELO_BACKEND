import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# === Configuración de conexión ===
usuario = "postgres"
contrasena = "Fabio123"
host = "localhost"
puerto = "5432"
basedatos = "modelo_empresas"
tabla = "empresa_datos"


engine = create_engine(f"postgresql+psycopg2://{usuario}:{contrasena}@{host}:{puerto}/{basedatos}")

df = pd.read_sql_table(tabla, con=engine)
print(f" Datos cargados desde PostgreSQL: {df.shape[0]} filas, {df.shape[1]} columnas")

# === Columnas numericas donde se revisarán outliers ===
cols_financieras = [
    'activo_corriente', 'capital_invertido', 'gastos_financieros',
    'gastos_por_prestamos', 'deudas_corto_plazo', 'patrimonio_empresa',
    'ganancia_operativa', 'total_activo', 'deudas_largo_plazo',
    'total_pasivo_y_patrimonio', 'ventas_netas'
]

# === Tratamiento de outliers ===
for col in cols_financieras:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {outliers} outliers detectados")
    df[col] = np.clip(df[col], lower, upper)
df['gastos_por_prestamos'] = np.where(df['gastos_por_prestamos'] < 0, 0, df['gastos_por_prestamos'])

# === Guardar tabla actualizada ===
df.to_sql(tabla, con=engine, if_exists='replace', index=False)
print(f"Tabla '{tabla}' actualizada con valores ajustados.")

