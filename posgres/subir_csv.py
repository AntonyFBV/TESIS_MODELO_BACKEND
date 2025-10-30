import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.types import Boolean, Integer, Float, String

# Configura tus credenciales de PostgreSQL
usuario = "postgres"
contrasena = "Fabio123"
host = "localhost"
puerto = "5432"
basedatos = "modelo_empresas"

# Crear conexión al motor de PostgreSQL
engine = create_engine(f"postgresql+psycopg2://{usuario}:{contrasena}@{host}:{puerto}/{basedatos}")

# Leer tu CSV limpio
df = pd.read_csv("dataset_Db.csv", sep=';')

# Reemplaza NaN por None (PostgreSQL no acepta NaN)
df = df.where(pd.notnull(df), None)

# Subir a PostgreSQL — crea la tabla automáticamente
df.to_sql(
    "empresa_datos",
    con=engine,
    if_exists="replace",  # usa "append" si quieres agregar sin borrar lo anterior
    index=False
)

print("✅ CSV subido correctamente a PostgreSQL.")
