import pandas as pd
from .features import COLUMNAS


def preparar_input(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y alinea el DataFrame al orden de columnas del modelo."""
    df = df.fillna(0)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    return df.reindex(columns=COLUMNAS, fill_value=0)
