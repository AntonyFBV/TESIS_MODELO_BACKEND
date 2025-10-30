import pandas as pd
import numpy as np

# ========================================
# 1. LECTURA DE DATOS
# ========================================
print("=" * 60)
print("📖 PASO 1: Leyendo archivo Excel...")
print("=" * 60)

archivo = "Base2020_2019_2018_2017.xlsx"

# Leer las 6 hojas (2015–2020)
df_2020 = pd.read_excel(archivo, sheet_name="Información 2020")
df_2019 = pd.read_excel(archivo, sheet_name="Información 2019")
df_2018 = pd.read_excel(archivo, sheet_name="Información 2018")
df_2017 = pd.read_excel(archivo, sheet_name="Información 2017")
df_2016 = pd.read_excel(archivo, sheet_name="Información 2016")
df_2015 = pd.read_excel(archivo, sheet_name="Información 2015")

# Limpiar nombres de columnas
for df in [df_2020, df_2019, df_2018, df_2017, df_2016, df_2015]:
    df.columns = df.columns.str.strip()

print(f"✓ Información 2020: {df_2020.shape[0]:,} filas × {df_2020.shape[1]} columnas")
print(f"✓ Información 2019: {df_2019.shape[0]:,} filas × {df_2019.shape[1]} columnas")
print(f"✓ Información 2018: {df_2018.shape[0]:,} filas × {df_2018.shape[1]} columnas")
print(f"✓ Información 2017: {df_2017.shape[0]:,} filas × {df_2017.shape[1]} columnas")
print(f"✓ Información 2016: {df_2016.shape[0]:,} filas × {df_2016.shape[1]} columnas")
print(f"✓ Información 2015: {df_2015.shape[0]:,} filas × {df_2015.shape[1]} columnas")

# Convertir IRUC a texto
for df in [df_2020, df_2019, df_2018, df_2017, df_2016, df_2015]:
    df["IRUC"] = df["IRUC"].astype(str)

# ========================================
# 2. AGREGAR COLUMNA DE AÑO
# ========================================
print("\n" + "=" * 60)
print("📅 PASO 2: Agregando columna 'Año'...")
print("=" * 60)

df_2020['Año'] = 2020
df_2019['Año'] = 2019
df_2018['Año'] = 2018
df_2017['Año'] = 2017
df_2016['Año'] = 2016
df_2015['Año'] = 2015

print("✓ Columna 'Año' agregada correctamente")

# ========================================
# 3. VERIFICAR ESTRUCTURA
# ========================================
print("\n" + "=" * 60)
print("🔍 PASO 3: Verificando estructura de columnas...")
print("=" * 60)

cols_2020 = set(df_2020.columns) - {'Año'}
cols_2019 = set(df_2019.columns) - {'Año'}
cols_2018 = set(df_2018.columns) - {'Año'}
cols_2017 = set(df_2017.columns) - {'Año'}
cols_2016 = set(df_2016.columns) - {'Año'}
cols_2015 = set(df_2015.columns) - {'Año'}

todas_iguales = (cols_2020 == cols_2019 == cols_2018 ==
                 cols_2017 == cols_2016 == cols_2015)

if todas_iguales:
    print(f"✓ Todas las hojas tienen las mismas {len(cols_2020)} columnas")
else:
    print("⚠️  Las hojas tienen columnas diferentes")
    print(f"  2020: {len(cols_2020)} columnas")
    print(f"  2019: {len(cols_2019)} columnas")
    print(f"  2018: {len(cols_2018)} columnas")
    print(f"  2017: {len(cols_2017)} columnas")
    print(f"  2016: {len(cols_2016)} columnas")
    print(f"  2015: {len(cols_2015)} columnas")

# ========================================
# 4. UNIFICAR COLUMNAS
# ========================================
print("\n" + "=" * 60)
print("🔧 PASO 4: Unificando estructura...")
print("=" * 60)

todas_columnas = sorted(list(cols_2020 | cols_2019 | cols_2018 | cols_2017 | cols_2016 | cols_2015))
print(f"✓ Total de columnas únicas: {len(todas_columnas)}")

columnas_categoricas = [
    'ACTIVIDADECONOMICA', 'MERCADERIAPRINCIPAL', 'Actividad economica',
    'Categoria', 'Tipo de Establecimiento', 'Condicion',
    'Departamento', 'Provincia', 'Distrito', 'canal_dominante'
]

def estandarizar_df(df, columnas_requeridas):
    for col in columnas_requeridas:
        if col not in df.columns:
            if col in columnas_categoricas:
                df[col] = 'NO APLICA'
            else:
                df[col] = 0
    cols_ordenadas = ['IRUC'] + [c for c in columnas_requeridas if c != 'IRUC'] + ['Año']
    return df[cols_ordenadas]

df_2020 = estandarizar_df(df_2020, todas_columnas)
df_2019 = estandarizar_df(df_2019, todas_columnas)
df_2018 = estandarizar_df(df_2018, todas_columnas)
df_2017 = estandarizar_df(df_2017, todas_columnas)
df_2016 = estandarizar_df(df_2016, todas_columnas)
df_2015 = estandarizar_df(df_2015, todas_columnas)

print(f"✓ Todos los DataFrames estandarizados ({len(todas_columnas)+2} columnas con IRUC y Año)")

# ========================================
# 5. CREAR PANEL LONGITUDINAL
# ========================================
print("\n" + "=" * 60)
print("🔗 PASO 5: Creando panel longitudinal...")
print("=" * 60)

panel = pd.concat([df_2015, df_2016, df_2017, df_2018, df_2019, df_2020], ignore_index=True)
panel = panel.sort_values(['IRUC', 'Año']).reset_index(drop=True)

print(f"✓ Panel creado exitosamente:")
print(f"  • Total de registros: {panel.shape[0]:,}")
print(f"  • Total de columnas: {panel.shape[1]:,}")
print(f"  • Empresas únicas: {panel['IRUC'].nunique():,}")

# ========================================
# 6. CREAR VARIABLE OBJETIVO
# ========================================
print("\n" + "=" * 60)
print("🎯 PASO 6: Creando variable objetivo 'Sobrevive_2años'...")
print("=" * 60)

empresas_años = panel.groupby('IRUC')['Año'].apply(set).to_dict()

def sobrevive_2anios(row):
    iruc = row['IRUC']
    anio_actual = row['Año']
    anio_futuro = anio_actual + 2
    if iruc in empresas_años and anio_futuro in empresas_años[iruc]:
        return 1
    else:
        return 0

panel['Sobrevive_2años'] = panel.apply(sobrevive_2anios, axis=1)

tasa_supervivencia = panel['Sobrevive_2años'].mean() * 100
print(f"✓ Variable objetivo creada exitosamente ({tasa_supervivencia:.1f}% sobreviven 2 años)")

# ========================================
# 7. GUARDAR DATASET COMPLETO
# ========================================
print("\n" + "=" * 60)
print("💾 PASO 7: Guardando dataset completo...")
print("=" * 60)

panel.to_csv("dataset_panel_completo.csv", index=False)
print(f"✓ dataset_panel_completo.csv guardado correctamente")
print(f"  ({panel.shape[0]:,} filas × {panel.shape[1]:,} columnas)")

# ========================================
# 8. RESUMEN FINAL
# ========================================
print("\n" + "=" * 60)
print("📋 RESUMEN FINAL")
print("=" * 60)

print(f"✅ Panel longitudinal creado exitosamente:")
print(f"  • Años incluidos: 2015–2020 (6 años)")
print(f"  • Total de registros: {panel.shape[0]:,}")
print(f"  • Total de variables: {panel.shape[1]:,}")
print(f"  • Empresas únicas: {panel['IRUC'].nunique():,}")

cols_preview = ['IRUC', 'Año', 'VENTAS', 'Totalempleados', 'canal_dominante', 
                'nro_canales_usados', 'Categoria', 'Sobrevive_2años']
cols_disponibles = [c for c in cols_preview if c in panel.columns]
print("\n📄 Vista previa del dataset:")
print(panel[cols_disponibles].head(5).to_string(index=False))

print("\n" + "=" * 60)
print("✅ PROCESO COMPLETADO - DATASET COMPLETO LISTO")
print("=" * 60)
