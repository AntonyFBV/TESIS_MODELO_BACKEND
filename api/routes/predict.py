import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from api.core.model_loader import get_model
from api.db.database import get_db
from api.db.models import Predictions, Recommendation
from api.ml.predictor import generar_recomendacion
from api.schemas.input import InputData

router = APIRouter()

# provincia, latitud y longitud son metadata de BD, no features del modelo
_EXCLUIR_DEL_MODELO = {"provincia", "latitud", "longitud"}


def _nivel_riesgo(prob_si: float) -> str:
    pct = prob_si * 100
    if pct <= 29:
        return "alto"
    elif pct <= 70:
        return "medio"
    return "bajo"


def _to_model_df(data: InputData) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in data.dict().items() if k not in _EXCLUIR_DEL_MODELO}])


# ─── PREDICT ───────────────────────────────────────────────────────────────────
@router.post("/predict")
def predict(data: InputData):
    pred = get_model().predict(_to_model_df(data))
    return {"prediction": int(pred[0])}


# ─── RECOMENDAR ────────────────────────────────────────────────────────────────
@router.post("/recomendar")
def recomendar(data: InputData):
    model = get_model()
    df = _to_model_df(data)
    proba = model.predict_proba(df)[0]
    prob_no, prob_si = float(proba[0]), float(proba[1])

    return {
        "probabilidades": {
            "sobrevive": round(prob_si * 100, 2),
            "no_sobrevive": round(prob_no * 100, 2),
        },
        "nivel_riesgo": _nivel_riesgo(prob_si),
        "recomendaciones": generar_recomendacion(df, idx=0, probabilidad=prob_si),
    }


# ─── PREDICT + GUARDAR ─────────────────────────────────────────────────────────
@router.post("/predict_save")
async def predict_save(data: InputData, user_id: int, db: AsyncSession = Depends(get_db)):
    model = get_model()
    df = _to_model_df(data)
    proba = model.predict_proba(df)[0]
    prob_si = float(proba[1])
    prediction_value = int(model.predict(df)[0])
    nivel = _nivel_riesgo(prob_si)

    # ─── guardar predicción ────────────────────────────────────────────────────
    new_prediction = Predictions(
        user_id=user_id,
        **data.dict(),
        prediction_result=prediction_value,
        nivel_riesgo=nivel,
    )
    db.add(new_prediction)
    await db.flush()  # genera el id sin cerrar la transacción

    # ─── guardar recomendaciones ───────────────────────────────────────────────
    resultado = generar_recomendacion(df, idx=0, probabilidad=prob_si)
    for rec in resultado["recomendaciones"]:
        db.add(Recommendation(
            prediction_id=new_prediction.id,
            variable=rec["variable"],
            valor_actual=str(rec["valor_actual"]),
            impacto=rec["impacto"],
            impacto_pct=rec["impacto_pct"],
            recomendacion=rec["recomendacion"],
        ))

    await db.commit()
    await db.refresh(new_prediction)

    return {"message": "Predicción guardada correctamente", "prediction_id": new_prediction.id}


# ─── HISTORIAL DE PREDICCIONES ─────────────────────────────────────────────────
@router.get("/predictions/{user_id}")
async def get_predictions(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Predictions)
        .where(Predictions.user_id == user_id)
        .options(selectinload(Predictions.recommendations))
        .order_by(Predictions.date_created.desc())
    )
    records = result.scalars().all()

    if not records:
        return {"message": "No hay predicciones registradas para este usuario"}

    return [
        {
            "id": p.id,
            "date_created": p.date_created,
            "provincia": p.provincia,
            "latitud": p.latitud,
            "longitud": p.longitud,
            "prediction_result": p.prediction_result,
            "nivel_riesgo": p.nivel_riesgo,
            "probabilidad_sobrevive": None,  # no se guarda, viene del modelo
            "log_capital": p.log_capital,
            "log_activos": p.log_activos,
            "total_patrimonio": p.total_patrimonio,
            "obligaciones_financieras": p.obligaciones_financieras,
            "log_ventas": p.log_ventas,
            "ganancias_netas": p.ganancias_netas,
            "margen_utilidad": p.margen_utilidad,
            "empleados": p.empleados,
            "publicidad": p.publicidad,
            "tiene_software": p.tiene_software,
            "cantidad_de_canales": p.cantidad_de_canales,
            "mostrador": p.mostrador,
            "comercio_electronico": p.comercio_electronico,
            "correo_catalogo_televentas": p.correo_catalogo_televentas,
            "telefono": p.telefono,
            "domicilio": p.domicilio,
            "ferias": p.ferias,
            "comisionistas": p.comisionistas,
            "maquinas_expendedoras": p.maquinas_expendedoras,
            "otros": p.otros,
            "recomendaciones": [
                {
                    "variable": r.variable,
                    "valor_actual": r.valor_actual,
                    "impacto": r.impacto,
                    "impacto_pct": r.impacto_pct,
                    "recomendacion": r.recomendacion,
                }
                for r in p.recommendations
            ],
        }
        for p in records
    ]