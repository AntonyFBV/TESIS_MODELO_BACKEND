# entrenamiento_avanzado_summary.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random
import warnings
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# === 0️⃣ Seed ===
SEED = 41
np.random.seed(SEED)
random.seed(SEED)

# === 1️⃣ Load data ===
print("🔄 Loading data from PostgreSQL...")
engine = create_engine("postgresql+psycopg2://postgres:Fabio123@localhost:5432/modelo_empresas")
df = pd.read_sql("SELECT * FROM public.empresa_datos", engine)
print(f"Dataset shape: {df.shape}")

# === 2️⃣ Variables ===
X = df.drop('sobrevive_2años', axis=1)
y = df['sobrevive_2años']

# === 3️⃣ Split data ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_test, X_eval, y_test, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)
print(f"Training: {X_train.shape[0]} | Test: {X_test.shape[0]} | Eval: {X_eval.shape[0]}")

# === 4️⃣ SMOTE ===
smote = SMOTE(random_state=SEED)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# === 5️⃣ ISSA function ===
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

# === 6️⃣ Scaling for logistic regression and SVM ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)
X_eval_scaled = scaler.transform(X_eval)

# === 7️⃣ Models & hyperparameter bounds ===
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

    if nombre not in ["Logistic Regression", "SVM"]:
        best_params["random_state"] = SEED

    modelo_final = modelo_class(**best_params)

    # === Entrenamiento ===
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

    # === Métricas ===
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

        # === 📈 Curva de Aprendizaje (Train vs Test vs Eval) ===
    print(f"📊 Generando curva de aprendizaje (Train/Test/Eval) para {nombre}...")

    # Seleccionar datos según si el modelo requiere escalado
    if nombre in ["SVM", "Logistic Regression"]:
        X_curve_train = X_train_scaled
        X_curve_test = X_test_scaled
        X_curve_eval = X_eval_scaled
    else:
        X_curve_train = X_train_bal
        X_curve_test = X_test
        X_curve_eval = X_eval

    y_curve_train = y_train_bal
    y_curve_test = y_test
    y_curve_eval = y_eval

    # Definir tamaños de entrenamiento (10% a 100%)
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores, test_scores, eval_scores = [], [], []

    for size in train_sizes:
        # Usar fracción del conjunto de entrenamiento
        n_samples = int(size * X_curve_train.shape[0])
        X_sub = X_curve_train[:n_samples]
        y_sub = y_curve_train[:n_samples]

        # Entrenar modelo temporal
        modelo_temp = modelo_class(**best_params)
        modelo_temp.fit(X_sub, y_sub)

        # Predicciones
        y_pred_train_sub = modelo_temp.predict(X_sub)
        y_pred_test_sub = modelo_temp.predict(X_curve_test)
        y_pred_eval_sub = modelo_temp.predict(X_curve_eval)

        # F1 Score en los tres conjuntos
        train_scores.append(f1_score(y_sub, y_pred_train_sub))
        test_scores.append(f1_score(y_curve_test, y_pred_test_sub))
        eval_scores.append(f1_score(y_curve_eval, y_pred_eval_sub))

    # Graficar resultados
    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes * 100, train_scores, 'o-', label="Entrenamiento")
    plt.plot(train_sizes * 100, test_scores, 'o-', label="Testeo")
    plt.plot(train_sizes * 100, eval_scores, 'o-', label="Evaluación")
    plt.title(f"Curva de Aprendizaje — {nombre}")
    plt.xlabel("Porcentaje de conjunto de entrenamiento utilizado (%)")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"learning_curve_{nombre.replace(' ', '_')}_3sets.png", dpi=300)
    plt.close()
    print(f"✅ Guardada: learning_curve_{nombre.replace(' ', '_')}_3sets.png")



# === 9️⃣ Final summary ===
summary_df = pd.DataFrame(summary).sort_values(by="Eval F1", ascending=False)
print("\n=== FINAL MODEL COMPARISON (TRAIN / TEST / EVAL) ===")
print(summary_df)

summary_df.to_csv("model_comparison_metrics.csv", index=False)
print("\n✅ Comparison metrics saved to 'model_comparison_metrics.csv'")
