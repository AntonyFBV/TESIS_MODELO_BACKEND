import xgboost as xgb
from .config import MODEL_PATH

# ─── CARGA ÚNICA DEL MODELO ────────────────────────────────────────────────────
_model: xgb.XGBClassifier | None = None


def get_model() -> xgb.XGBClassifier:
    global _model
    if _model is None:
        _model = xgb.XGBClassifier()
        _model.load_model(MODEL_PATH)
    return _model
