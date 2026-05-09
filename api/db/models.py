from datetime import datetime, timezone, timedelta
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from .database import Base


# ─── USER ──────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    predictions = relationship("Predictions", back_populates="user")


# ─── PREDICTIONS ───────────────────────────────────────────────────────────────
class Predictions(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    provincia = Column(Integer)

    # ─── columnas nuevas (modelo actualizado) ──────────────────────────────────
    log_capital = Column(Float)
    log_activos = Column(Float)
    total_patrimonio = Column(Integer)
    obligaciones_financieras = Column(Boolean)
    log_ventas = Column(Float)
    ganancias_netas = Column(Float)
    margen_utilidad = Column(Float)
    empleados = Column(Integer)
    publicidad = Column(Boolean)
    tiene_software = Column(Boolean)
    cantidad_de_canales = Column(Integer)
    mostrador = Column(Boolean)
    comercio_electronico = Column(Boolean)
    correo_catalogo_televentas = Column(Boolean)
    telefono = Column(Boolean)
    domicilio = Column(Boolean)
    ferias = Column(Boolean)
    comisionistas = Column(Boolean)
    maquinas_expendedoras = Column(Boolean)
    otros = Column(Boolean)

    # ─── columnas anteriores (compatibilidad) ──────────────────────────────────
    capital_invertido = Column(Integer)
    gastos_por_prestamos = Column(Float)
    deudas_corto_plazo = Column(Integer)
    patrimonio_empresa = Column(Float)
    total_activo = Column(Float)
    numero_empleados = Column(Integer)
    ventas_netas = Column(Integer)
    canal_principal = Column(Integer)
    cantidad_canales_venta = Column(Integer)
    vende_por_catalogo = Column(Boolean)
    vende_con_comisionistas = Column(Boolean)
    vende_a_domicilio = Column(Boolean)
    vende_en_linea = Column(Boolean)
    vende_en_ferias = Column(Boolean)
    vende_en_maquinas_venta = Column(Boolean)
    vende_en_tienda = Column(Boolean)
    otros_medios_venta = Column(Boolean)
    vende_por_telefono = Column(Boolean)
    uso_de_software = Column(Boolean)

    # ─── resultado ─────────────────────────────────────────────────────────────
    prediction_result = Column(Integer)
    nivel_riesgo = Column(String)
    latitud = Column(Float)
    longitud = Column(Float)
    date_created = Column(DateTime, default=lambda: datetime.now(timezone(timedelta(hours=-5))).replace(tzinfo=None))

    user = relationship("User", back_populates="predictions")
    recommendations = relationship("Recommendation", back_populates="prediction", cascade="all, delete")


# ─── RECOMMENDATIONS ───────────────────────────────────────────────────────────
class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    variable = Column(String)
    valor_actual = Column(String)
    impacto = Column(Float)
    impacto_pct = Column(Float)
    recomendacion = Column(Text)

    prediction = relationship("Predictions", back_populates="recommendations")