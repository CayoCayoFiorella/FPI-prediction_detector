"""
Quick Fix: Verificar y corregir normalización del target FVC
"""

import pandas as pd
import numpy as np

# Cargar datos originales
df = pd.read_csv('data/raw/train.csv')

print("="*70)
print("🔍 ANÁLISIS DE NORMALIZACIÓN DE FVC")
print("="*70)

# Estadísticas de FVC
fvc_values = df['FVC'].values
print(f"\n📊 Estadísticas de FVC en train.csv:")
print(f"   Min:    {fvc_values.min():.0f} ml")
print(f"   Max:    {fvc_values.max():.0f} ml")
print(f"   Mean:   {fvc_values.mean():.0f} ml")
print(f"   Median: {np.median(fvc_values):.0f} ml")
print(f"   Std:    {fvc_values.std():.0f} ml")

# Verificar tus predicciones
df_pred = pd.read_csv('predictions_analysis.csv')
pred_values = df_pred['prediction'].values

print(f"\n📊 Estadísticas de PREDICCIONES:")
print(f"   Min:    {pred_values.min():.0f} ml")
print(f"   Max:    {pred_values.max():.0f} ml")
print(f"   Mean:   {pred_values.mean():.0f} ml")
print(f"   Median: {np.median(pred_values):.0f} ml")
print(f"   Std:    {pred_values.std():.0f} ml")

# Comparar
print(f"\n🔍 DIAGNÓSTICO:")
print(f"   Ratio Mean (pred/real): {pred_values.mean() / fvc_values.mean():.3f}")

if pred_values.mean() < fvc_values.mean() * 0.8:
    print("\n❌ PROBLEMA IDENTIFICADO: Modelo SUBPREDICE sistemáticamente")
    print("\n💡 POSIBLES CAUSAS:")
    print("   1. Target FVC no se desnormalizó después de predicción")
    print("   2. Modelo entrenado con FVC normalizado pero evaluado sin desnormalizar")
    print("   3. Loss function sesgada hacia valores bajos")
    
    # Calcular factor de corrección
    factor = fvc_values.mean() / pred_values.mean()
    print(f"\n🔧 SOLUCIÓN RÁPIDA (para probar):")
    print(f"   Multiplicar predicciones por factor: {factor:.3f}")
    
    # Aplicar corrección
    pred_corrected = pred_values * factor
    errors_corrected = df_pred['target'].values - pred_corrected
    mae_corrected = np.abs(errors_corrected).mean()
    
    print(f"\n📊 Métricas CORREGIDAS (experimental):")
    print(f"   MAE corregido: {mae_corrected:.0f} ml")
    
    ss_res = np.sum(errors_corrected**2)
    ss_tot = np.sum((df_pred['target'].values - df_pred['target'].mean())**2)
    r2_corrected = 1 - (ss_res / ss_tot)
    print(f"   R² corregido:  {r2_corrected:.4f}")

else:
    print("\n✅ Rango de predicciones parece correcto")
    print("   El problema está en la ARQUITECTURA del modelo, no en normalización")

print("\n" + "="*70)
EOF