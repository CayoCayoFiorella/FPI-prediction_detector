"""
Script para corregir predicciones normalizadas.
Desnormaliza multiplicando por 5000.
"""

import pandas as pd
import numpy as np

print("="*70)
print("🔧 CORRECCIÓN DE PREDICCIONES - Desnormalización")
print("="*70)

# Cargar predicciones actuales
df = pd.read_csv('predictions_analysis.csv')

print(f"\n📊 Predicciones originales (normalizadas):")
print(f"   Min:    {df['prediction'].min():.2f}")
print(f"   Max:    {df['prediction'].max():.2f}")
print(f"   Mean:   {df['prediction'].mean():.2f}")

# Desnormalizar (multiplicar por 5000)
df['prediction_fixed'] = df['prediction'] * 5000.0
df['error_fixed'] = df['target'] - df['prediction_fixed']
df['abs_error_fixed'] = np.abs(df['error_fixed'])

print(f"\n📊 Predicciones desnormalizadas:")
print(f"   Min:    {df['prediction_fixed'].min():.2f} ml")
print(f"   Max:    {df['prediction_fixed'].max():.2f} ml")
print(f"   Mean:   {df['prediction_fixed'].mean():.2f} ml")

# Calcular métricas corregidas
mae = df['abs_error_fixed'].mean()
rmse = np.sqrt((df['error_fixed']**2).mean())
mape = (df['abs_error_fixed'] / df['target']).mean() * 100

ss_res = np.sum(df['error_fixed']**2)
ss_tot = np.sum((df['target'] - df['target'].mean())**2)
r2 = 1 - (ss_res / ss_tot)

print("\n" + "="*70)
print("📊 MÉTRICAS CORREGIDAS")
print("="*70)
print(f"\n   MAE:  {mae:.2f} ml   (era 1574 ml)")
print(f"   RMSE: {rmse:.2f} ml")
print(f"   MAPE: {mape:.2f}%    (era 57%)")
print(f"   R²:   {r2:.4f}      (era -3.38)")

# Comparación
print("\n" + "="*70)
print("📈 MEJORA")
print("="*70)
print(f"   MAE:  1574 ml  →  {mae:.0f} ml  ({(1574-mae)/1574*100:.1f}% mejor)")
print(f"   R²:   -3.38    →  {r2:.2f}      ({'✅ POSITIVO' if r2>0 else '⚠️ Aún negativo'})")

if r2 > 0:
    print("\n✅ ¡ÉXITO! Tu modelo SÍ funciona correctamente.")
    print("   Solo necesitabas desnormalizar las predicciones.")
else:
    print("\n⚠️ R² aún negativo. Necesitas entrenar más épocas.")

# Guardar
df.to_csv('predictions_fixed.csv', index=False)
print(f"\n💾 Guardado en: predictions_fixed.csv")
print("="*70)