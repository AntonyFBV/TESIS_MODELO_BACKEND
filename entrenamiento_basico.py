# entrenamiento_basico_csv.py
# -*- coding: utf-8 -*-

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === 1. Cargar datos desde PostgreSQL ===
print("🔄 Cargando datos desde PostgreSQL...")
engine = create_engine("postgresql+psycopg2://postgres:Fabio123@localhost:5432/modelo_empresas")
df = pd.read_sql("SELECT * FROM public.empresa_datos", engine)

print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

# --- Nuevo bloque: revisar la columna objetivo ---
print("🔍 Conteo original de 'sobrevive_2años':")
print(df['sobrevive_2años'].value_counts())

# Convertir a 0/1 si no lo está
df['sobrevive_2años'] = df['sobrevive_2años'].replace({True: 1, False: 0, 'true': 1, 'false': 0})

print("\n🔢 Después de convertir a 0/1:")
print(df['sobrevive_2años'].value_counts())
# --- fin del bloque ---


# === 2. Definir variables ===
X = df.drop('sobrevive_2años', axis=1)
y = df['sobrevive_2años']

# === 3. Dividir dataset ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_test, X_eval, y_test, y_eval = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"📊 Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Eval: {X_eval.shape[0]}")

# === 4. Escalar para algunos modelos ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_eval_scaled = scaler.transform(X_eval)

# === 5. Definir modelos básicos ===
modelos = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "Logistic Regression": LogisticRegression(max_iter=5000, solver='lbfgs', random_state=42),
    "SVM": SVC(random_state=42)
}

# === 6. Entrenar y evaluar ===
resultados = []

for nombre, modelo in modelos.items():
    print(f"\n🚀 Entrenando modelo: {nombre}...")
    
    # Escalar si es necesario
    if nombre in ["Logistic Regression", "SVM"]:
        modelo.fit(X_train_scaled, y_train)
        y_pred_train = modelo.predict(X_train_scaled)
        y_pred_test = modelo.predict(X_test_scaled)
        y_pred_eval = modelo.predict(X_eval_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)
        y_pred_eval = modelo.predict(X_eval)
    
    resultados.append({
        "Model": nombre,
        "Train Accuracy": accuracy_score(y_train, y_pred_train),
        "Train Precision": precision_score(y_train, y_pred_train, zero_division=0),
        "Train Recall": recall_score(y_train, y_pred_train),
        "Train F1": f1_score(y_train, y_pred_train),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Test Precision": precision_score(y_test, y_pred_test, zero_division=0),
        "Test Recall": recall_score(y_test, y_pred_test),
        "Test F1": f1_score(y_test, y_pred_test),
        "Eval Accuracy": accuracy_score(y_eval, y_pred_eval),
        "Eval Precision": precision_score(y_eval, y_pred_eval, zero_division=0),
        "Eval Recall": recall_score(y_eval, y_pred_eval),
        "Eval F1": f1_score(y_eval, y_pred_eval)
    })

# === 7. Guardar resultados en CSV ===
resultados_df = pd.DataFrame(resultados).sort_values(by="Test F1", ascending=False)
resultados_df.to_csv("model_basic_metrics.csv", index=False)
print("\n✅ Resultados guardados en 'model_basic_metrics.csv'")
print(resultados_df)
