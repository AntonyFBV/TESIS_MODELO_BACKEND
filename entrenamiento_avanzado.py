# entrenamiento_avanzado_summary.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import warnings
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

SEED = 41
np.random.seed(SEED)
random.seed(SEED)

# === Cargar data ===
print("Loading data from PostgreSQL...")
engine = create_engine("postgresql+psycopg2://postgres:Fabio123@localhost:5432/modelo_empresas")
df = pd.read_sql("SELECT * FROM public.empresa_datos", engine)
print(f"Dataset shape: {df.shape}")

# === Variables ===
X = df.drop('sobrevive_2años', axis=1)
y = df['sobrevive_2años']

# === Division de data ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_test, X_eval, y_test, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)
print(f"Training: {X_train.shape[0]} | Test: {X_test.shape[0]} | Eval: {X_eval.shape[0]}")

# === SMOTE ===
smote = SMOTE(random_state=SEED)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# === ISSA ===
def ISSA(model_class, X, y, param_bounds, n_iter=50, cv_folds=5, seed=SEED):
    best_params = None
    best_score = -np.inf
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for _ in range(n_iter):
        params = {}
        for p, v in param_bounds.items():
            val = random.uniform(v[0], v[1])
            if any(x in p for x in ["depth", "estimators", "leaves", "split"]):
                val = int(round(val))
            params[p] = val
        try:
            model = model_class(**params, random_state=seed)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        except Exception:
            continue
    return best_params, best_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)
X_eval_scaled = scaler.transform(X_eval)

# === Modelos usados ===
modelos = {
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "LightGBM": LGBMClassifier,
    "Logistic Regression": LogisticRegression,
    "SVM": SVC
}

param_bounds_all = {
    "Decision Tree": {"max_depth": [2, 15], "min_samples_split": [2, 10]},
    "Random Forest": {"n_estimators": [50, 300], "max_depth": [3, 15]},
    "XGBoost": {"n_estimators": [50, 300], "learning_rate": [0.01, 0.3], "max_depth": [3, 10]},
    "LightGBM": {"n_estimators": [50, 300], "learning_rate": [0.01, 0.3], "num_leaves": [20, 80]},
    "Logistic Regression": {"C": [0.01, 10]},
    "SVM": {"C": [0.1, 10]}
}

# === 8️⃣ Train, optimize and evaluate ===
summary = []

for nombre, modelo_class in modelos.items():
    print(f"\n🔍 Optimizing {nombre} with ISSA + SMOTE + CV...")

    bounds = param_bounds_all[nombre]
    X_opt = X_train_scaled if nombre in ["Logistic Regression", "SVM"] else X_train_bal
    y_opt = y_train_bal

    best_params, best_cv_score = ISSA(modelo_class, X_opt, y_opt, bounds, n_iter=50, cv_folds=5, seed=SEED)

    # Random state for reproducibility
    if nombre not in ["Logistic Regression", "SVM"]:
        best_params["random_state"] = SEED

    modelo_final = modelo_class(**best_params)

    # Train
    if nombre in ["Logistic Regression", "SVM"]:
        modelo_final.fit(X_train_scaled, y_train_bal)
        y_pred_train = modelo_final.predict(X_train_scaled)
        y_pred_test = modelo_final.predict(X_test_scaled)
        y_pred_eval = modelo_final.predict(X_eval_scaled)
    else:
        modelo_final.fit(X_train_bal, y_train_bal)
        y_pred_train = modelo_final.predict(X_train_bal)
        y_pred_test = modelo_final.predict(X_test)
        y_pred_eval = modelo_final.predict(X_eval)

    # Store metrics
    summary.append({
        "Model": nombre,
        "Best Params": best_params,
        "CV_F1": best_cv_score,
        "Train Accuracy": accuracy_score(y_train_bal, y_pred_train),
        "Train Precision": precision_score(y_train_bal, y_pred_train),
        "Train Recall": recall_score(y_train_bal, y_pred_train),
        "Train F1": f1_score(y_train_bal, y_pred_train),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Test Precision": precision_score(y_test, y_pred_test),
        "Test Recall": recall_score(y_test, y_pred_test),
        "Test F1": f1_score(y_test, y_pred_test),
        "Eval Accuracy": accuracy_score(y_eval, y_pred_eval),
        "Eval Precision": precision_score(y_eval, y_pred_eval),
        "Eval Recall": recall_score(y_eval, y_pred_eval),
        "Eval F1": f1_score(y_eval, y_pred_eval)
    })

# === 9️⃣ Final summary ===
summary_df = pd.DataFrame(summary).sort_values(by="Eval F1", ascending=False)
print("\n=== FINAL MODEL COMPARISON (TRAIN / TEST / EVAL) ===")
print(summary_df)

# Optionally save to CSV
summary_df.to_csv("model_comparison_metrics.csv", index=False)
print("\n✅ Comparison metrics saved to 'model_comparison_metrics.csv'")
