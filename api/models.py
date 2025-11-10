from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    predictions = relationship("Predictions", back_populates="user")

class Predictions(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    capital_invertido = Column(Integer)
    provincia = Column(Integer)
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
    publicidad = Column(Boolean)
    uso_de_software = Column(Boolean)

    prediction_result=Column(Integer)
    date_created= Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")