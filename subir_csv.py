import pandas as pd
from sqlalchemy import create_engine

# Configura tus credenciales de PostgreSQL
usuario = "postgres"
contrasena = "Fabio123"
host = "localhost"
puerto = "5432"
basedatos = "modelo_empresas"

# Crear conexión al motor de PostgreSQL
engine = create_engine(f"postgresql+psycopg2://{usuario}:{contrasena}@{host}:{puerto}/{basedatos}")

# Leer tu CSV limpio
df = pd.read_csv("DATASET_FINAL.csv", sep=';')

# Subir a PostgreSQL — crea la tabla automáticamente
df.to_sql("empresa_datos", con=engine, if_exists="replace", index=False)

print("✅ CSV subido correctamente a PostgreSQL.")
