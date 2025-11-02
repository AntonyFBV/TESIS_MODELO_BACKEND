# api/main.py
from fastapi import FastAPI,Depends,HTTPException,status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .db import SessionLocal
from .models import User
from .auth import verify_password, get_password_hash, create_access_token
import pandas as pd
import os
import xgboost as xgb  # ✅ usar xgboost directamente

# 👇 Importar la función del recomendador
from .recomendador_xgboost import generar_recomendacion


# Crear la instancia de FastAPI
app = FastAPI(title="XGBoost Prediction API")

async def get_db():
    async with SessionLocal() as session:
        yield session

# Habilitar CORS
origins = ["http://localhost:4200"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Cargar modelo XGBoost en formato JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "xgboost_model_final.json")

model = xgb.XGBClassifier()
model.load_model(model_path)

# ✅ Fin del cambio

# Definir los campos esperados
class InputData(BaseModel):
    capital_invertido: int
    provincia: int
    gastos_por_prestamos: float
    deudas_corto_plazo: int
    patrimonio_empresa: float
    total_activo: float
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
    publicidad:bool
    uso_de_software: bool

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}


@app.post("/recomendar")
def recomendar(data: InputData):
    df = pd.DataFrame([data.dict()])

    prob = float(model.predict_proba(df)[0][1])
    recomendaciones = generar_recomendacion(df, idx=0, probabilidad=prob)

    return {
        "probabilidad_sobrevivir": round(prob, 4),
        "recomendaciones": recomendaciones
    }

# Endpoint Registro
@app.post("/register")
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == data.username))
    existing_user = result.scalars().first()

    if existing_user:
        raise HTTPException(status_code=400, detail="El usuario ya existe")

    hashed_pw = get_password_hash(data.password[:72])  # corta si pasa el límite bcrypt
    new_user = User(username=data.username, email=data.email, hashed_password=hashed_pw)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return {"msg": "Usuario registrado exitosamente", "user": new_user.username}


# Endpoint Login
@app.post("/login")
async def login(username: str, password: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalars().first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer","user_id":user.id}

# Endpoint Listar usuario
@app.get("/user/{user_id}")
async def get_user_by_id(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email

    }

from fastapi import Body

class UserUpdateRequest(BaseModel):
    username: str | None = None
    email: str | None = None
    password: str | None = None

@app.put("/user/{user_id}")
@app.patch("/user/{user_id}")
@app.put("/user/{user_id}")
@app.patch("/user/{user_id}")
async def update_user(
    user_id: int,
    data: UserUpdateRequest = Body(...),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # Validaciones de email y username
    if data.email and data.email != user.email:
        email_check = await db.execute(select(User).where(User.email == data.email))
        if email_check.scalars().first():
            raise HTTPException(status_code=400, detail="El correo ya está registrado")

    if data.username and data.username != user.username:
        username_check = await db.execute(select(User).where(User.username == data.username))
        if username_check.scalars().first():
            raise HTTPException(status_code=400, detail="El nombre de usuario ya está en uso")

    # Actualizaciones
    if data.username:
        user.username = data.username
    if data.email:
        user.email = data.email
    if data.password:
        user.hashed_password = get_password_hash(data.password[:72])

    await db.commit()
    await db.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "message": "Usuario actualizado correctamente"
    }

# 👇 Agregar al final del archivo
from .db import engine, Base
import asyncio

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        print("🔄 Verificando tablas en la base de datos...")
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tablas listas en la base de datos")