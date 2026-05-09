from pydantic import BaseModel


# ─── INPUT DEL MODELO ──────────────────────────────────────────────────────────
class InputData(BaseModel):
    log_capital: float
    log_activos: float
    total_patrimonio: int
    obligaciones_financieras: bool
    log_ventas: float
    ganancias_netas: float
    margen_utilidad: float
    empleados: int
    publicidad: bool
    tiene_software: bool
    cantidad_de_canales: int
    mostrador: bool
    comercio_electronico: bool
    correo_catalogo_televentas: bool
    telefono: bool
    domicilio: bool
    ferias: bool
    comisionistas: bool
    maquinas_expendedoras: bool
    otros: bool
    # ─── extra (no van al modelo, sí a BD) ────────────────────────────────────
    provincia: int
    latitud: float
    longitud: float


# ─── AUTH ──────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class UserUpdateRequest(BaseModel):
    username: str | None = None
    email: str | None = None
    password: str | None = None