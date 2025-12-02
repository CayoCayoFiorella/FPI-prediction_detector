"""
Genera reporte visual completo del entrenamiento desde MLflow.
Incluye gráficas de métricas, análisis de convergencia y resumen textual.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def generate_report(experiment_name="Default"):
    """
    Genera reporte completo con gráficas del entrenamiento.
    
    Args:
        experiment_name: Nombre del experimento en MLflow
    """
    
    print("="*70)
    print(f"📊 GENERANDO REPORTE DE ENTRENAMIENTO")
    print("="*70)
    print(f"Experimento: {experiment_name}\n")
    
    # Configurar MLflow
    mlflow.set_tracking_uri("mlruns")
    
    # Obtener experimento
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ Experimento '{experiment_name}' no encontrado")
            print("\nExperimentos disponibles:")
            for exp in mlflow.search_experiments():
                print(f"  - {exp.name}")
            return
    except Exception as e:
        print(f"❌ Error al acceder a MLflow: {e}")
        return
    
    # Buscar runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Filtrar runs completados
    finished_runs = runs[runs['status'] == 'FINISHED']
    
    if len(finished_runs) == 0:
        print("❌ No hay runs completados exitosamente")
        print(f"\nRuns disponibles: {len(runs)}")
        print(f"  - Completados: 0")
        print(f"  - Fallidos: {len(runs[runs['status'] == 'FAILED'])}")
        print(f"  - En ejecución: {len(runs[runs['status'] == 'RUNNING'])}")
        return
    
    # Tomar el último run completado (más reciente)
    last_run = finished_runs.iloc[0]
    run_id = last_run['run_id']
    
    print(f"✅ Run seleccionado: {run_id[:12]}...")
    print(f"📅 Fecha: {last_run['start_time']}")
    print(f"⏱️ Estado: {last_run['status']}\n")
    
    # Obtener métricas por época
    client = mlflow.tracking.MlflowClient()
    metrics_history = {}
    
    metric_names = [
        'train_mae', 'val_mae', 
        'train_loss', 'val_loss', 
        'train_rmse', 'val_rmse', 
        'train_r2', 'val_r2',
        'train_mse', 'val_mse',
        'train_mape', 'val_mape',
        'train_ccc', 'val_ccc'
    ]
    
    print("📈 Cargando métricas...")
    for metric_name in metric_names:
        try:
            metric_history = client.get_metric_history(run_id, metric_name)
            if len(metric_history) > 0:
                metrics_history[metric_name] = [(m.step, m.value) for m in metric_history]
                print(f"   ✓ {metric_name}: {len(metric_history)} puntos")
        except Exception as e:
            pass
    
    if len(metrics_history) == 0:
        print("❌ No se pudieron cargar métricas del run")
        return
    
    print(f"\n✅ {len(metrics_history)} métricas cargadas\n")
    
    # ==================== CREAR FIGURA ====================
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
    
    # Colores
    color_train = '#2E86AB'
    color_val = '#A23B72'
    
    # ==================== 1. MAE ====================
    if 'train_mae' in metrics_history and 'val_mae' in metrics_history:
        ax = fig.add_subplot(gs[0, 0])
        train_mae = metrics_history['train_mae']
        val_mae = metrics_history['val_mae']
        
        epochs_train = [x[0] for x in train_mae]
        values_train = [x[1] for x in train_mae]
        epochs_val = [x[0] for x in val_mae]
        values_val = [x[1] for x in val_mae]
        
        ax.plot(epochs_train, values_train, label='Train', 
                linewidth=2.5, color=color_train, marker='o', markersize=3)
        ax.plot(epochs_val, values_val, label='Validation', 
                linewidth=2.5, color=color_val, marker='s', markersize=3)
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('MAE (ml)', fontsize=11, fontweight='bold')
        ax.set_title('Mean Absolute Error', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # ==================== 2. LOSS ====================
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        ax = fig.add_subplot(gs[0, 1])
        train_loss = metrics_history['train_loss']
        val_loss = metrics_history['val_loss']
        
        epochs_train = [x[0] for x in train_loss]
        values_train = [x[1] for x in train_loss]
        epochs_val = [x[0] for x in val_loss]
        values_val = [x[1] for x in val_loss]
        
        ax.plot(epochs_train, values_train, label='Train', 
                linewidth=2.5, color=color_train, marker='o', markersize=3)
        ax.plot(epochs_val, values_val, label='Validation', 
                linewidth=2.5, color=color_val, marker='s', markersize=3)
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
        ax.set_title('Loss Function', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # ==================== 3. RMSE ====================
    if 'train_rmse' in metrics_history and 'val_rmse' in metrics_history:
        ax = fig.add_subplot(gs[0, 2])
        train_rmse = metrics_history['train_rmse']
        val_rmse = metrics_history['val_rmse']
        
        epochs_train = [x[0] for x in train_rmse]
        values_train = [x[1] for x in train_rmse]
        epochs_val = [x[0] for x in val_rmse]
        values_val = [x[1] for x in val_rmse]
        
        ax.plot(epochs_train, values_train, label='Train', 
                linewidth=2.5, color=color_train, marker='o', markersize=3)
        ax.plot(epochs_val, values_val, label='Validation', 
                linewidth=2.5, color=color_val, marker='s', markersize=3)
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE (ml)', fontsize=11, fontweight='bold')
        ax.set_title('Root Mean Squared Error', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # ==================== 4. R² ====================
    if 'train_r2' in metrics_history and 'val_r2' in metrics_history:
        ax = fig.add_subplot(gs[1, 0])
        train_r2 = metrics_history['train_r2']
        val_r2 = metrics_history['val_r2']
        
        epochs_train = [x[0] for x in train_r2]
        values_train = [x[1] for x in train_r2]
        epochs_val = [x[0] for x in val_r2]
        values_val = [x[1] for x in val_r2]
        
        ax.plot(epochs_train, values_train, label='Train', 
                linewidth=2.5, color=color_train, marker='o', markersize=3)
        ax.plot(epochs_val, values_val, label='Validation', 
                linewidth=2.5, color=color_val, marker='s', markersize=3)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline')
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('R²', fontsize=11, fontweight='bold')
        ax.set_title('R² Score (Coefficient of Determination)', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # ==================== 5. MAPE ====================
    if 'train_mape' in metrics_history and 'val_mape' in metrics_history:
        ax = fig.add_subplot(gs[1, 1])
        train_mape = metrics_history['train_mape']
        val_mape = metrics_history['val_mape']
        
        epochs_train = [x[0] for x in train_mape]
        values_train = [x[1] for x in train_mape]
        epochs_val = [x[0] for x in val_mape]
        values_val = [x[1] for x in val_mape]
        
        ax.plot(epochs_train, values_train, label='Train', 
                linewidth=2.5, color=color_train, marker='o', markersize=3)
        ax.plot(epochs_val, values_val, label='Validation', 
                linewidth=2.5, color=color_val, marker='s', markersize=3)
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('MAPE (%)', fontsize=11, fontweight='bold')
        ax.set_title('Mean Absolute Percentage Error', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # ==================== 6. CCC ====================
    if 'train_ccc' in metrics_history and 'val_ccc' in metrics_history:
        ax = fig.add_subplot(gs[1, 2])
        train_ccc = metrics_history['train_ccc']
        val_ccc = metrics_history['val_ccc']
        
        epochs_train = [x[0] for x in train_ccc]
        values_train = [x[1] for x in train_ccc]
        epochs_val = [x[0] for x in val_ccc]
        values_val = [x[1] for x in val_ccc]
        
        ax.plot(epochs_train, values_train, label='Train', 
                linewidth=2.5, color=color_train, marker='o', markersize=3)
        ax.plot(epochs_val, values_val, label='Validation', 
                linewidth=2.5, color=color_val, marker='s', markersize=3)
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('CCC', fontsize=11, fontweight='bold')
        ax.set_title('Concordance Correlation Coefficient', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # ==================== 7. MEJORA RELATIVA ====================
    if 'val_mae' in metrics_history:
        ax = fig.add_subplot(gs[2, 0])
        val_mae = metrics_history['val_mae']
        
        epochs = [x[0] for x in val_mae]
        values = [x[1] for x in val_mae]
        
        initial_value = values[0]
        improvement = [(initial_value - v) / initial_value * 100 for v in values]
        
        ax.plot(epochs, improvement, linewidth=3, color='#06A77D', marker='D', markersize=4)
        ax.fill_between(epochs, 0, improvement, alpha=0.3, color='#06A77D')
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mejora (%)', fontsize=11, fontweight='bold')
        ax.set_title('Mejora Relativa de Validation MAE', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Anotar mejora final
        final_improvement = improvement[-1]
        ax.annotate(f'{final_improvement:.1f}%', 
                   xy=(epochs[-1], final_improvement),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # ==================== 8. CONVERGENCIA (LOSS) ====================
    if 'train_loss' in metrics_history and 'val_loss' in metrics_history:
        ax = fig.add_subplot(gs[2, 1])
        
        train_loss = metrics_history['train_loss']
        val_loss = metrics_history['val_loss']
        
        epochs_train = [x[0] for x in train_loss]
        values_train = [x[1] for x in train_loss]
        epochs_val = [x[0] for x in val_loss]
        values_val = [x[1] for x in val_loss]
        
        # Calcular gap entre train y val
        gap = [v - t for v, t in zip(values_val, values_train)]
        
        ax.fill_between(epochs_train, values_train, values_val, alpha=0.3, color='orange', label='Gap Train-Val')
        ax.plot(epochs_train, values_train, linewidth=2, color=color_train, label='Train')
        ax.plot(epochs_val, values_val, linewidth=2, color=color_val, label='Validation')
        
        ax.set_xlabel('Época', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax.set_title('Análisis de Overfitting (Loss Gap)', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # ==================== 9. TABLA DE MÉTRICAS FINALES ====================
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    # Calcular métricas finales
    final_epoch = len(metrics_history.get('train_mae', [(0, 0)]))
    
    metrics_text = "MÉTRICAS FINALES\n" + "="*40 + "\n\n"
    metrics_text += f"Época: {final_epoch}\n\n"
    
    metrics_text += "VALIDACIÓN:\n"
    if 'val_mae' in metrics_history:
        metrics_text += f"  MAE:   {metrics_history['val_mae'][-1][1]:.2f} ml\n"
    if 'val_rmse' in metrics_history:
        metrics_text += f"  RMSE:  {metrics_history['val_rmse'][-1][1]:.2f} ml\n"
    if 'val_r2' in metrics_history:
        metrics_text += f"  R²:    {metrics_history['val_r2'][-1][1]:.4f}\n"
    if 'val_mape' in metrics_history:
        metrics_text += f"  MAPE:  {metrics_history['val_mape'][-1][1]:.2f}%\n"
    if 'val_ccc' in metrics_history:
        metrics_text += f"  CCC:   {metrics_history['val_ccc'][-1][1]:.4f}\n"
    
    metrics_text += "\nENTRENAMIENTO:\n"
    if 'train_mae' in metrics_history:
        metrics_text += f"  MAE:   {metrics_history['train_mae'][-1][1]:.2f} ml\n"
    if 'train_rmse' in metrics_history:
        metrics_text += f"  RMSE:  {metrics_history['train_rmse'][-1][1]:.2f} ml\n"
    if 'train_r2' in metrics_history:
        metrics_text += f"  R²:    {metrics_history['train_r2'][-1][1]:.4f}\n"
    
    ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    # ==================== 10. RESUMEN TEXTUAL ====================
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')
    
    # Calcular estadísticas
    if 'val_mae' in metrics_history:
        val_mae_values = [x[1] for x in metrics_history['val_mae']]
        best_mae = min(val_mae_values)
        best_epoch = val_mae_values.index(best_mae) + 1
        initial_mae = val_mae_values[0]
        final_mae = val_mae_values[-1]
        improvement_pct = (initial_mae - final_mae) / initial_mae * 100
    else:
        best_mae = best_epoch = initial_mae = final_mae = improvement_pct = 0
    
    # Obtener valores finales de forma segura
    final_r2 = metrics_history['val_r2'][-1][1] if 'val_r2' in metrics_history else None
    final_ccc = metrics_history['val_ccc'][-1][1] if 'val_ccc' in metrics_history else None
    
    # Formatear valores
    r2_str = f"{final_r2:.4f}" if final_r2 is not None else "N/A"
    ccc_str = f"{final_ccc:.4f}" if final_ccc is not None else "N/A"
    batch_size = last_run.get('params.batch_size', 'N/A')
    lr = last_run.get('params.learning_rate', 'N/A')
    optimizer = last_run.get('params.optimizer', 'N/A')
    loss_fn = last_run.get('params.loss', 'N/A')
    
    summary_text = f"""
{'='*110}
RESUMEN EJECUTIVO DEL ENTRENAMIENTO
{'='*110}

Experimento: {experiment_name}          Run ID: {run_id[:16]}...          Fecha: {last_run['start_time'].strftime('%Y-%m-%d %H:%M')}

CONFIGURACIÓN:
  • Batch size: {batch_size}    • Learning rate: {lr}    • Optimizer: {optimizer}    • Loss: {loss_fn}

RESULTADOS:
  • Épocas completadas: {final_epoch}          • Mejor época: {best_epoch}          • Mejora total: {improvement_pct:.1f}% (MAE)
  
  • MAE inicial: {initial_mae:.2f} ml → MAE final: {final_mae:.2f} ml → Mejor MAE: {best_mae:.2f} ml (época {best_epoch})
  
  • R² final: {r2_str}          • CCC final: {ccc_str}

{'='*110}
    """
    
    ax.text(0.5, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Título general
    fig.suptitle(f'REPORTE COMPLETO DE ENTRENAMIENTO - Predicción de Progresión de FPI', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Guardar figura
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'training_report_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Reporte guardado en: {output_file}")
    
    # Guardar CSV con métricas
    if 'train_mae' in metrics_history:
        data = {'epoch': [x[0] for x in metrics_history['train_mae']]}
        
        for metric_name, values in metrics_history.items():
            data[metric_name] = [v[1] for v in values]
        
        df_metrics = pd.DataFrame(data)
        csv_file = f'training_metrics_{timestamp}.csv'
        df_metrics.to_csv(csv_file, index=False)
        print(f"✅ Métricas guardadas en: {csv_file}")
    
    # Mostrar figura
    plt.show()
    
    print("\n" + "="*70)
    print("✅ REPORTE GENERADO EXITOSAMENTE")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    # Permitir especificar experimento desde línea de comandos
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        experiment_name = "Default"
    
    print(f"\n🚀 Iniciando generación de reporte para: {experiment_name}\n")
    generate_report(experiment_name)