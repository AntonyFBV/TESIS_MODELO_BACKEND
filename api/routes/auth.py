from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from api.core.security import create_access_token, get_password_hash, verify_password
from api.db.database import get_db
from api.db.models import User
from api.schemas.input import RegisterRequest, UserUpdateRequest

router = APIRouter()


# ─── REGISTER ──────────────────────────────────────────────────────────────────
@router.post("/register")
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == data.username))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="El usuario ya existe")

    new_user = User(
        username=data.username,
        email=data.email,
        hashed_password=get_password_hash(data.password[:72]),
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return {"msg": "Usuario registrado exitosamente", "user": new_user.username}


# ─── LOGIN ─────────────────────────────────────────────────────────────────────
@router.post("/login")
async def login(username: str, password: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalars().first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}


# ─── GET USER ──────────────────────────────────────────────────────────────────
@router.get("/user/{user_id}")
async def get_user_by_id(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return {"id": user.id, "username": user.username, "email": user.email}


# ─── UPDATE USER ───────────────────────────────────────────────────────────────
@router.put("/user/{user_id}")
@router.patch("/user/{user_id}")
async def update_user(
    user_id: int,
    data: UserUpdateRequest = Body(...),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # Verificar contraseña actual para cualquier modificación
    if (
        data.username != user.username
        or data.email != user.email
        or data.password
    ):
        if not data.current_password:
            raise HTTPException(
                status_code=400,
                detail="Debe ingresar su contraseña actual"
            )

        if not verify_password(
            data.current_password,
            user.hashed_password
        ):
            raise HTTPException(
                status_code=400,
                detail="La contraseña actual es incorrecta"
            )

    if data.email and data.email != user.email:
        check = await db.execute(
            select(User).where(User.email == data.email)
        )
        if check.scalars().first():
            raise HTTPException(
                status_code=400,
                detail="El correo ya está registrado"
            )

    if data.username and data.username != user.username:
        check = await db.execute(
            select(User).where(User.username == data.username)
        )
        if check.scalars().first():
            raise HTTPException(
                status_code=400,
                detail="El nombre de usuario ya está en uso"
            )

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
        "message": "Usuario actualizado correctamente",
    }
