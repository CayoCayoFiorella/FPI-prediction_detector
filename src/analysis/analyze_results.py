"""
Analiza los resultados de experimentos guardados en MLflow.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import mlflow
import pandas as pd

def analyze_experiments(experiment_name="Default"):  # ← CAMBIO AQUÍ
    """Analiza todos los experimentos de MLflow."""
    
    # Configurar MLflow
    mlflow.set_tracking_uri("mlruns")
    
    # Obtener experimento
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ No se encontró el experimento '{experiment_name}'")
            print("Experimentos disponibles:")
            for exp in mlflow.search_experiments():
                print(f"  - {exp.name}")
            return
        
        experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"❌ Error al conectar con MLflow: {e}")
        return
    
    # Buscar runs
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    
    if len(runs) == 0:
        print(f"❌ No se encontraron runs en el experimento '{experiment_name}'")
        return
    
    print("="*70)
    print(f"📊 ANÁLISIS DE EXPERIMENTOS - {experiment_name}")
    print("="*70)
    
    # Información general
    print(f"\n📈 Total de runs: {len(runs)}")
    print(f"📅 Fecha más reciente: {runs['start_time'].max()}")
    
    # Filtrar solo runs exitosos
    finished_runs = runs[runs['status'] == 'FINISHED']
    print(f"✅ Runs completados: {len(finished_runs)}")
    print(f"❌ Runs fallidos: {len(runs[runs['status'] == 'FAILED'])}")
    
    if len(finished_runs) == 0:
        print("\n⚠️ No hay runs completados exitosamente")
        return
    
    # Mostrar información de runs
    print("\n" + "="*70)
    print("🔹 RUNS COMPLETADOS:")
    print("="*70)
    
    for idx, row in finished_runs.iterrows():
        print(f"\n📌 Run {idx + 1}:")
        print(f"   ID: {row['run_id'][:12]}...")
        print(f"   Fecha: {row['start_time']}")
        print(f"   Duración: {row.get('end_time', 'N/A')}")
        
        # Métricas finales
        metrics_cols = [col for col in row.index if col.startswith('metrics.')]
        if metrics_cols:
            print(f"   Métricas finales:")
            for col in metrics_cols:
                metric_name = col.replace('metrics.', '')
                value = row[col]
                if pd.notna(value):
                    print(f"      {metric_name}: {value:.4f}")
    
    # Mejor modelo
    if 'metrics.val_mae' in finished_runs.columns:
        best_run = finished_runs.loc[finished_runs['metrics.val_mae'].idxmin()]
        
        print("\n" + "="*70)
        print("🏆 MEJOR MODELO (menor Validation MAE):")
        print("="*70)
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  Fecha: {best_run['start_time']}")
        
        print(f"\n📊 Métricas de Validación:")
        val_metrics = {
            'val_mae': 'MAE',
            'val_rmse': 'RMSE',
            'val_r2': 'R²',
            'val_mape': 'MAPE',
            'val_ccc': 'CCC',
            'val_loss': 'Loss'
        }
        
        for metric_key, metric_label in val_metrics.items():
            full_key = f'metrics.{metric_key}'
            if full_key in best_run.index and pd.notna(best_run[full_key]):
                print(f"  {metric_label:6s}: {best_run[full_key]:>10.4f}")
        
        print(f"\n⚙️ Hiperparámetros:")
        param_keys = ['batch_size', 'learning_rate', 'optimizer', 'num_epochs', 'loss']
        for param in param_keys:
            full_key = f'params.{param}'
            if full_key in best_run.index and pd.notna(best_run[full_key]):
                print(f"  {param}: {best_run[full_key]}")
    
    # Guardar resumen
    summary_file = 'experiments_summary.csv'
    finished_runs.to_csv(summary_file, index=False)
    print(f"\n💾 Resumen guardado en: {summary_file}")
    
    print("\n" + "="*70)
    print("✅ Análisis completado")
    print("="*70)
    print("\n💡 Tip: Ejecuta 'mlflow ui --backend-store-uri mlruns' para ver gráficas")


if __name__ == "__main__":
    analyze_experiments("Default")  # Buscar en Default