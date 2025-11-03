"""
Utilidades para el proceso de entrenamiento:
- Early stopping
- Checkpointing
- Configuración de semillas
- Schedulers
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional


def set_seed(seed: int):
    """
    Establece semilla para reproducibilidad.
    
    Args:
        seed: Valor de semilla
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Para mayor determinismo (puede afectar performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Semilla establecida: {seed}")


class EarlyStopping:
    """
    Early stopping para detener entrenamiento cuando no hay mejora.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Número de épocas sin mejora antes de detener
            min_delta: Cambio mínimo para considerar mejora
            mode: 'min' para minimizar, 'max' para maximizar
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < (best - min_delta)
        else:
            self.monitor_op = lambda current, best: current > (best + min_delta)
    
    def __call__(self, current_score: float) -> bool:
        """
        Evalúa si debe detenerse el entrenamiento.
        
        Args:
            current_score: Métrica actual a monitorear
        
        Returns:
            True si debe detenerse, False en caso contrario
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        if self.monitor_op(current_score, self.best_score):
            # Hay mejora
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            # No hay mejora
            self.counter += 1
            print(f"⏰ EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🛑 Early stopping activado!")
                return True
        
        return False


class ModelCheckpoint:
    """
    Guarda checkpoints del modelo.
    """
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        filename_prefix: str = 'model'
    ):
        """
        Args:
            save_dir: Directorio donde guardar checkpoints
            monitor: Métrica a monitorear
            mode: 'min' o 'max'
            save_best_only: Si solo guardar el mejor modelo
            filename_prefix: Prefijo para nombre de archivo
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.filename_prefix = filename_prefix
        self.best_score = None
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda current, best: current > best
            self.best_score = float('-inf')
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        is_best: bool = False
    ):
        """
        Guarda checkpoint.
        
        Args:
            model: Modelo a guardar
            optimizer: Optimizador
            epoch: Época actual
            metrics: Diccionario de métricas
            is_best: Si es el mejor modelo hasta ahora
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Guardar checkpoint regular
        if not self.save_best_only:
            filename = self.save_dir / f"{self.filename_prefix}_epoch_{epoch}.pth"
            torch.save(checkpoint, filename)
            print(f"💾 Checkpoint guardado: {filename}")
        
        # Guardar mejor modelo
        if is_best:
            best_filename = self.save_dir / f"{self.filename_prefix}_best.pth"
            torch.save(checkpoint, best_filename)
            print(f"🏆 Mejor modelo guardado: {best_filename}")
    
    def __call__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict
    ):
        """
        Evalúa y guarda checkpoint si corresponde.
        """
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            print(f"⚠️ Métrica '{self.monitor}' no encontrada en metrics")
            return
        
        is_best = self.monitor_op(current_score, self.best_score)
        
        if is_best:
            self.best_score = current_score
            print(f"✨ Nueva mejor métrica {self.monitor}: {current_score:.4f}")
        
        self.save(model, optimizer, epoch, metrics, is_best)


def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Crea optimizador según configuración.
    
    Args:
        model: Modelo PyTorch
        config: Diccionario de configuración
    
    Returns:
        Optimizador
    """
    optimizer_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizador no soportado: {optimizer_name}")
    
    print(f"🔧 Optimizador: {optimizer_name.upper()}, LR: {lr}, Weight Decay: {weight_decay}")
    return optimizer


def get_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """
    Crea scheduler según configuración.
    
    Args:
        optimizer: Optimizador
        config: Diccionario de configuración del scheduler
    
    Returns:
        Scheduler o None
    """
    if not config or config.get('type') is None:
        return None
    
    scheduler_type = config.get('type', '').lower()
    
    if scheduler_type == 'cosine':
        T_max = config.get('T_max', 50)
        eta_min = config.get('min_lr', 0.000001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        print(f"📊 Scheduler: CosineAnnealing, T_max={T_max}, eta_min={eta_min}")
    
    elif scheduler_type == 'reduce_on_plateau':
        patience = config.get('patience', 5)
        factor = config.get('factor', 0.5)
        min_lr = config.get('min_lr', 0.000001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=patience,
            factor=factor,
            min_lr=min_lr
        )
        print(f"📊 Scheduler: ReduceLROnPlateau, patience={patience}, factor={factor}")
    
    elif scheduler_type == 'step':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        print(f"📊 Scheduler: StepLR, step_size={step_size}, gamma={gamma}")
    
    else:
        print(f"⚠️ Scheduler '{scheduler_type}' no reconocido")
        return None
    
    return scheduler


def get_loss_function(loss_name: str):
    """
    Retorna función de loss según nombre.
    
    Args:
        loss_name: Nombre del loss ('mse', 'mae', 'huber')
    
    Returns:
        Función de loss
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'mse':
        return torch.nn.MSELoss()
    elif loss_name == 'mae' or loss_name == 'l1':
        return torch.nn.L1Loss()
    elif loss_name == 'huber':
        return torch.nn.HuberLoss()
    else:
        raise ValueError(f"Loss function no soportada: {loss_name}")


if __name__ == "__main__":
    print("🧪 Testing training utilities...\n")
    
    # Test set_seed
    set_seed(42)
    
    # Test EarlyStopping
    print("\n--- Testing EarlyStopping ---")
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
    
    test_scores = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84]
    for i, score in enumerate(test_scores):
        print(f"Epoch {i+1}, Score: {score}")
        should_stop = early_stopping(score)
        if should_stop:
            print(f"Detenido en época {i+1}")
            break
    
    print("\n✅ Utilities test exitoso!")