"""
Métricas de evaluación para predicción de FVC.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List
import torch


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de regresión.
    
    Args:
        y_true: Valores verdaderos (N,)
        y_pred: Valores predichos (N,)
    
    Returns:
        Diccionario con métricas
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    # Evitar división por cero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Coeficiente de Correlación de Concordancia (CCC).
    Métrica común en análisis de concordancia médica.
    
    CCC = 2 * pearson_corr * std(y_true) * std(y_pred) / 
          (var(y_true) + var(y_pred) + (mean(y_true) - mean(y_pred))^2)
    """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # Covarianza
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    # CCC
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    if denominator == 0:
        return 0.0
    
    ccc = numerator / denominator
    return float(ccc)


class MetricsTracker:
    """Clase para rastrear métricas durante el entrenamiento."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reinicia el tracker."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """
        Actualiza el tracker con nuevos valores.
        
        Args:
            predictions: Tensor de predicciones
            targets: Tensor de targets
            loss: Valor de loss
        """
        # Convertir a numpy y aplanar
        preds = predictions.detach().cpu().numpy()
        tgts = targets.detach().cpu().numpy()
        
        # Asegurar que sean 1D
        if preds.ndim == 0:
            preds = np.array([preds.item()])
        else:
            preds = preds.flatten()
        
        if tgts.ndim == 0:
            tgts = np.array([tgts.item()])
        else:
            tgts = tgts.flatten()
        
        # Actualizar listas
        self.predictions.extend(preds.tolist())
        self.targets.extend(tgts.tolist())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """
        Calcula todas las métricas acumuladas.
        
        Returns:
            Diccionario con métricas
        """
        if len(self.predictions) == 0:
            return {}
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        metrics = calculate_metrics(y_true, y_pred)
        metrics['loss'] = float(np.mean(self.losses))
        metrics['ccc'] = concordance_correlation_coefficient(y_true, y_pred)
        
        return metrics
    
    def get_predictions(self):
        """Retorna predicciones y targets como arrays."""
        return np.array(self.predictions), np.array(self.targets)


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Imprime métricas de forma legible.
    
    Args:
        metrics: Diccionario de métricas
        prefix: Prefijo para el print (ej. "Train", "Val")
    """
    print(f"\n{'='*50}")
    print(f"{prefix} Metrics:")
    print(f"{'='*50}")
    
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():>10}: {value:>10.4f}")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Test de métricas
    print("🧪 Testing metrics...\n")
    
    # Datos dummy
    y_true = np.array([3000, 3200, 2800, 3100, 2900])
    y_pred = np.array([3050, 3150, 2850, 3000, 2950])
    
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, prefix="Test")
    
    # Test de CCC
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    print(f"CCC: {ccc:.4f}")
    
    # Test de MetricsTracker
    tracker = MetricsTracker()
    tracker.update(torch.tensor(y_pred[:2]), torch.tensor(y_true[:2]), 100.0)
    tracker.update(torch.tensor(y_pred[2:]), torch.tensor(y_true[2:]), 80.0)
    
    final_metrics = tracker.compute()
    print_metrics(final_metrics, prefix="Tracked")
    
    print("✅ Metrics test exitoso!")