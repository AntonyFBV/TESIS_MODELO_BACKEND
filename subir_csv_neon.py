import pandas as pd
from sqlalchemy import create_engine

# ─── Credenciales Neon ─────────────────────────────────────────────────────
engine = create_engine(
    "postgresql+psycopg2://neondb_owner:npg_m0CpE2SabQoi@ep-delicate-darkness-a8xf4x2f-pooler.eastus2.azure.neon.tech/neondb?sslmode=require",
    connect_args={"sslmode": "require"}
)

# ─── Cargar CSV ────────────────────────────────────────────────────────────
df = pd.read_csv("dataset.csv", sep=';')
df = df.where(pd.notnull(df), None)

# ─── Subir a Neon ──────────────────────────────────────────────────────────
df.to_sql(
    "empresa_datos",
    con=engine,
    if_exists="replace",
    index=False
)

print("CSV subido correctamente a Neon.")