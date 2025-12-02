"""
Análisis detallado de errores del modelo.
Identifica patrones en predicciones incorrectas.
"""

import sys
import os
from pathlib import Path

# Configurar paths correctamente
project_root = Path(__file__).parent.parent.parent  # Desde src/analysis/ subir 2 niveles
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

print(f"📁 Project root: {project_root}")
print(f"📁 Working directory: {os.getcwd()}")

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
import yaml

# Importar módulos del proyecto con rutas absolutas
try:
    from src.dataset.osic_dataset import OSICDataset
    from src.preprocessing.preprocessing import CTPreprocessor
    from src.models.fusion_model import MultimodalFusionModel
    print("✅ Módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("\nIntentando importación alternativa...")
    # Alternativa: añadir src al path y usar imports relativos
    import dataset.osic_dataset as dataset_module
    import preprocessing.preprocessing as preproc_module
    import models.fusion_model as models_module
    OSICDataset = dataset_module.OSICDataset
    CTPreprocessor = preproc_module.CTPreprocessor
    MultimodalFusionModel = models_module.MultimodalFusionModel


def analyze_errors():
    """Analiza dónde y por qué el modelo falla."""
    
    print("\n" + "="*70)
    print("🔍 ANÁLISIS DE ERRORES DEL MODELO")
    print("="*70)
    
    # Verificar que existe el checkpoint
    checkpoint_path = Path('checkpoints/model_best.pth')
    if not checkpoint_path.exists():
        print(f"\n❌ No se encuentra el checkpoint: {checkpoint_path}")
        
        # Buscar checkpoints disponibles
        ckpt_dir = Path('checkpoints')
        if ckpt_dir.exists():
            available = list(ckpt_dir.glob('*.pth'))
            if available:
                print("\n📦 Checkpoints disponibles:")
                for f in available:
                    print(f"  - {f.name}")
                # Usar el primero disponible
                checkpoint_path = available[0]
                print(f"\n✅ Usando: {checkpoint_path.name}")
            else:
                print("❌ No hay checkpoints .pth disponibles")
                return
        else:
            print("❌ No existe la carpeta 'checkpoints'")
            return
    
    # Cargar configuración
    config_path = Path('experiments/configs/config.yaml')
    if not config_path.exists():
        print(f"❌ No se encuentra configuración: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n📊 Configuración cargada")
    print(f"   Dataset: {config['data']['csv_train']}")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Preparar preprocessor
    print("\n🔧 Preparando preprocessor...")
    preproc = CTPreprocessor(
        target_spacing=tuple(config['preprocessing']['target_spacing']),
        target_size=tuple(config['preprocessing']['target_size']),
        lung_window=tuple(config['preprocessing']['lung_window']),
        clip_range=tuple(config['preprocessing']['clip_range'])
    )
    
    # Cargar dataset completo
    print("📦 Cargando dataset...")
    try:
        dataset = OSICDataset(
            csv_path=config['data']['csv_train'],
            dicom_root=config['data']['dicom_root_train'],
            preprocessor=preproc,
            use_cache=True,
            cache_dir=config['data']['cache_dir']
        )
        print(f"   ✅ Dataset cargado: {len(dataset)} muestras")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Usar un subset para análisis rápido
    analysis_size = min(300, len(dataset))  # Reducido a 300 para velocidad
    dataset_subset = Subset(dataset, range(analysis_size))
    print(f"   📊 Analizando primeras {analysis_size} muestras (reducido para velocidad)")
    
        # Cargar modelo
    print("\n🧠 Cargando modelo...")
    try:
        # PyTorch 2.9+ requiere weights_only=False
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Verificar tipo de objeto guardado
        if isinstance(loaded, dict):
            # Es un checkpoint dict
            print(f"   Checkpoint de época: {loaded.get('epoch', 'N/A')}")
            
            # Obtener dimensión clínica
            sample = dataset[0]
            clinical_dim = sample[1].shape[0]
            
            # Crear modelo
            model = MultimodalFusionModel(
                clinical_input_dim=clinical_dim,
                img_feature_dim=config['model']['img_feature_dim'],
                clinical_embedding_dim=config['model']['clinical_embedding_dim'],
                fusion_hidden_dims=config['model']['fusion_hidden_dims'],
                use_lightweight_cnn=config['model']['use_lightweight_cnn']
            )
            
            # Cargar pesos
            model.load_state_dict(loaded['model_state_dict'])
            
        else:
            # Es el modelo directamente (guardado por MLflow)
            print(f"   Tipo: Modelo guardado directamente por MLflow")
            model = loaded
        
        model.eval()
        print("   ✅ Modelo cargado exitosamente")
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Hacer predicciones
    print("\n📊 Generando predicciones...")
    predictions = []
    targets = []
    
    loader = DataLoader(dataset_subset, batch_size=1, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for idx, (image, clinical, weeks, target) in enumerate(loader):
            try:
                pred = model(image, clinical, weeks)
                predictions.append(pred.item())
                targets.append(target.item())
                
                if (idx + 1) % 50 == 0:
                    print(f"   Procesado: {idx + 1}/{len(loader)}")
            except Exception as e:
                print(f"   ⚠️ Error en muestra {idx}: {e}")
                continue
    
    if len(predictions) == 0:
        print("❌ No se generaron predicciones")
        return
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    errors = targets - predictions
    abs_errors = np.abs(errors)
    
    print(f"\n✅ {len(predictions)} predicciones generadas")
    
    # Calcular métricas
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(np.abs(errors / (targets + 1e-8))) * 100
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print("\n📈 MÉTRICAS CALCULADAS:")
    print(f"   MAE:  {mae:.2f} ml")
    print(f"   RMSE: {rmse:.2f} ml")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R²:   {r2:.4f}")
    
    # Crear figura de análisis
    print("\n📊 Generando visualizaciones...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('white')
    
    # 1. Scatter: Predicciones vs Real
    ax = axes[0, 0]
    ax.scatter(targets, predictions, alpha=0.6, s=30, c='blue', edgecolors='navy', linewidth=0.5)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Ideal', alpha=0.8)
    ax.set_xlabel('FVC Real (ml)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FVC Predicho (ml)', fontsize=12, fontweight='bold')
    ax.set_title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.0f} ml', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 2. Distribución de errores
    ax = axes[0, 1]
    ax.hist(errors, bins=40, alpha=0.75, edgecolor='black', color='skyblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2.5, label='Error = 0')
    ax.axvline(np.mean(errors), color='green', linestyle=':', linewidth=2, label=f'Media')
    ax.set_xlabel('Error (Real - Predicho) ml', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Errores', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Residual plot
    ax = axes[0, 2]
    ax.scatter(predictions, errors, alpha=0.6, s=30, c='purple')
    ax.axhline(0, color='red', linestyle='--', linewidth=2.5)
    ax.set_xlabel('FVC Predicho (ml)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuo (ml)', fontsize=12, fontweight='bold')
    ax.set_title('Gráfica de Residuos', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Error por rango de FVC
    ax = axes[1, 0]
    bins = np.linspace(targets.min(), targets.max(), 9)
    bin_indices = np.digitize(targets, bins)
    bin_errors = []
    bin_centers = []
    
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_errors.append(abs_errors[mask].mean())
            bin_centers.append((bins[i-1] + bins[i]) / 2)
    
    ax.bar(bin_centers, bin_errors, width=np.diff(bins)[0]*0.7, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Rango de FVC Real (ml)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE Promedio (ml)', fontsize=12, fontweight='bold')
    ax.set_title('Error por Rango de FVC', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Top 10 peores
    ax = axes[1, 1]
    ax.axis('off')
    
    worst_indices = np.argsort(abs_errors)[-10:][::-1]
    text = "🔴 TOP 10 PEORES PREDICCIONES\n" + "="*42 + "\n\n"
    for rank, idx in enumerate(worst_indices, 1):
        text += f"{rank:2d}. Real: {targets[idx]:>6.0f} │ Pred: {predictions[idx]:>6.0f} │ Err: {abs_errors[idx]:>6.0f}\n"
    
    ax.text(0.05, 0.5, text, fontsize=10, family='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='#FFE6E6', alpha=0.9))
    
    # 6. Estadísticas
    ax = axes[1, 2]
    ax.axis('off')
    
    pct_500 = (abs_errors < 500).sum() / len(abs_errors) * 100
    pct_1000 = (abs_errors < 1000).sum() / len(abs_errors) * 100
    pct_1500 = (abs_errors < 1500).sum() / len(abs_errors) * 100
    
    stats_text = "📊 ESTADÍSTICAS\n" + "="*30 + "\n\n"
    stats_text += f"Muestras: {len(predictions)}\n\n"
    stats_text += f"MAE:   {mae:>7.2f} ml\n"
    stats_text += f"RMSE:  {rmse:>7.2f} ml\n"
    stats_text += f"MAPE:  {mape:>7.2f} %\n"
    stats_text += f"R²:    {r2:>7.4f}\n\n"
    stats_text += f"Error < 500ml:  {pct_500:>5.1f}%\n"
    stats_text += f"Error < 1000ml: {pct_1000:>5.1f}%\n"
    stats_text += f"Error < 1500ml: {pct_1500:>5.1f}%"
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('ANÁLISIS DETALLADO DE ERRORES DEL MODELO', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = 'error_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Análisis guardado en: {output_file}")
    
    # Guardar CSV
    df_results = pd.DataFrame({
        'target': targets,
        'prediction': predictions,
        'error': errors,
        'abs_error': abs_errors
    })
    csv_file = 'predictions_analysis.csv'
    df_results.to_csv(csv_file, index=False)
    print(f"✅ Predicciones guardadas en: {csv_file}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    try:
        analyze_errors()
    except Exception as e:
        print(f"\n❌ ERROR GENERAL: {e}")
        import traceback
        traceback.print_exc()