# entrenamiento_xgboost_final.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import warnings
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Silenciar advertencias
warnings.filterwarnings('ignore')

# Fijar semillas
np.random.seed(41)
random.seed(41)

# === 1️⃣ Cargar datos ===
print("🔄 Cargando datos desde PostgreSQL...")
engine = create_engine("postgresql+psycopg2://postgres:Fabio123@localhost:5432/modelo_empresas")
df = pd.read_sql("SELECT * FROM public.empresa_datos", engine)
print(f"Dimensiones del dataset: {df.shape}")

# === 2️⃣ Definir variables ===
X = df.drop('sobrevive_2años', axis=1)
y = df['sobrevive_2años']

# === 3️⃣ División de datos ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_test, X_eval, y_test, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba: {X_test.shape[0]} muestras")
print(f"Evaluación: {X_eval.shape[0]} muestras")

# === 4️⃣ Balanceo con SMOTE ===
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"✅ Clases balanceadas en entrenamiento: {X_train_bal.shape[0]} muestras")

# === 5️⃣ Definir modelo con los mejores hiperparámetros ===
best_params = {
    'n_estimators': 203,
    'learning_rate': 0.2963,
    'max_depth': 8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

xgb_model = XGBClassifier(**best_params)

# === 6️⃣ Entrenar modelo final ===
xgb_model.fit(X_train_bal, y_train_bal)
print("\n✅ Entrenamiento completado")

# === 7️⃣ Evaluación en entrenamiento, test y evaluación ===
def evaluate_model(model, X, y, dataset_name="Dataset"):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"\n=== {dataset_name} Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    return acc, prec, rec, f1

# Entrenamiento
train_metrics = evaluate_model(xgb_model, X_train_bal, y_train_bal, "Training Set")

# Test
test_metrics = evaluate_model(xgb_model, X_test, y_test, "Test Set")

# Evaluación final
eval_metrics = evaluate_model(xgb_model, X_eval, y_eval, "Evaluation Set")

# === 8️⃣ Guardar modelo en formato oficial de XGBoost ===
xgb_model.save_model('xgboost_model_final.json')
print("\n✅ Modelo XGBoost guardado como 'xgboost_model_final.json'")

# === 9️⃣ Guardar métricas y parámetros ===
metrics = {
    'best_params': best_params,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics,
    'eval_metrics': eval_metrics
}

joblib.dump(metrics, 'xgboost_model_metrics.pkl')
print("✅ Métricas y parámetros guardados en 'xgboost_model_metrics.pkl'")
