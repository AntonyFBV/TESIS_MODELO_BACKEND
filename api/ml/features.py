# ─── COLUMNAS (orden exacto del modelo entrenado) ─────────────────────────────
# provincia NO va aquí — es metadata de BD, no feature del modelo
COLUMNAS = [
    "log_capital", "log_activos", "total_patrimonio", "obligaciones_financieras",
    "log_ventas", "ganancias_netas", "margen_utilidad", "empleados",
    "publicidad", "tiene_software", "cantidad_de_canales",
    "mostrador", "comercio_electronico", "correo_catalogo_televentas",
    "telefono", "domicilio", "ferias", "comisionistas",
    "maquinas_expendedoras", "otros",
]

# ─── SUBTIPOS ──────────────────────────────────────────────────────────────────
NUMERICAS = [
    "log_capital", "log_activos", "total_patrimonio",
    "log_ventas", "ganancias_netas", "margen_utilidad", "empleados",
]

CATEGORICAS = ["cantidad_de_canales"]

BOOLEANAS = [
    "obligaciones_financieras", "publicidad", "tiene_software",
    "mostrador", "comercio_electronico", "correo_catalogo_televentas",
    "telefono", "domicilio", "ferias", "comisionistas",
    "maquinas_expendedoras", "otros",
]

VARIABLES_RELEVANTES = NUMERICAS + CATEGORICAS + BOOLEANAS
