"""
Script principal de entrenamiento para el modelo de predicción de FPI.
Incluye:
- Carga de datos
- Entrenamiento con validación
- Early stopping
- Checkpointing
- Logging con MLflow
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml
import argparse
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from dataset.osic_dataset import OSICDataset
from preprocessing.preprocessing import CTPreprocessor
from models.fusion_model import MultimodalFusionModel, ClinicalOnlyModel
from evaluation.metrics import MetricsTracker, print_metrics
from training.utils import (
    set_seed, EarlyStopping, ModelCheckpoint,
    get_optimizer, get_scheduler, get_loss_function
)


class Trainer:
    """Clase principal para entrenamiento."""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Diccionario de configuración
        """
        self.config = config
        self.device = torch.device(config['experiment']['device'])
        
        # Configurar semilla
        set_seed(config['experiment']['seed'])
        
        # Preparar datos
        self._prepare_data()
        
        # Crear modelo
        self._create_model()
        
        # Configurar entrenamiento
        self._setup_training()
        
        # Configurar MLflow
        self._setup_mlflow()
    
    def _prepare_data(self):
        """Prepara datasets y dataloaders."""
        print("\n" + "="*60)
        print("📦 PREPARANDO DATOS")
        print("="*60)
        
        # Crear preprocessor
        preproc_config = self.config['preprocessing']
        self.preprocessor = CTPreprocessor(
            target_spacing=tuple(preproc_config['target_spacing']),
            target_size=tuple(preproc_config['target_size']),
            lung_window=tuple(preproc_config['lung_window']),
            clip_range=tuple(preproc_config['clip_range'])
        )
        
        # Cargar dataset completo
        data_config = self.config['data']
        full_dataset = OSICDataset(
            csv_path=data_config['csv_train'],
            dicom_root=data_config['dicom_root_train'],
            preprocessor=self.preprocessor,
            use_cache=data_config['use_cache'],
            cache_dir=data_config['cache_dir']
        )
        
        # Split train/val
        val_split = self.config['validation']['val_split']
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config['experiment']['seed'])
        )
        
        print(f"✅ Train samples: {len(self.train_dataset)}")
        print(f"✅ Val samples: {len(self.val_dataset)}")
        
        # Crear dataloaders
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['resources']['num_workers']
        pin_memory = self.config['resources']['pin_memory']
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        print(f"✅ Dataloaders creados (batch_size={batch_size})")
    
    def _create_model(self):
        """Crea el modelo según configuración."""
        print("\n" + "="*60)
        print("🧠 CREANDO MODELO")
        print("="*60)
        
        model_config = self.config['model']
        model_type = model_config['type']
        
        # Obtener dimensión de features clínicos del dataset
        sample = self.train_dataset[0]
        clinical_dim = sample[1].shape[0]  # clinical features
        
        if model_type == 'multimodal':
            self.model = MultimodalFusionModel(
                clinical_input_dim=clinical_dim,
                img_feature_dim=model_config['img_feature_dim'],
                clinical_embedding_dim=model_config['clinical_embedding_dim'],
                fusion_hidden_dims=model_config['fusion_hidden_dims'],
                use_lightweight_cnn=model_config['use_lightweight_cnn']
            )
        elif model_type == 'clinical_only':
            self.model = ClinicalOnlyModel(clinical_input_dim=clinical_dim)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        self.model.to(self.device)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Modelo: {model_type}")
        print(f"📊 Parámetros totales: {total_params:,}")
        print(f"📊 Parámetros entrenables: {trainable_params:,}")
    
    def _setup_training(self):
        """Configura componentes de entrenamiento."""
        print("\n" + "="*60)
        print("⚙️ CONFIGURANDO ENTRENAMIENTO")
        print("="*60)
        
        train_config = self.config['training']
        
        # Loss function
        loss_name = train_config['loss']
        self.criterion = get_loss_function(loss_name)
        print(f"✅ Loss: {loss_name.upper()}")
        
        # Optimizer
        self.optimizer = get_optimizer(self.model, train_config)
        
        # Scheduler
        scheduler_config = train_config.get('scheduler')
        self.scheduler = get_scheduler(self.optimizer, scheduler_config)
        
        # Early stopping
        es_config = train_config.get('early_stopping', {})
        if es_config.get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=es_config.get('patience', 10),
                min_delta=es_config.get('min_delta', 0.001),
                mode='min'
            )
            print(f"✅ Early Stopping: patience={es_config.get('patience', 10)}")
        else:
            self.early_stopping = None
        
        # Checkpoint
        ckpt_config = self.config['checkpoint']
        self.checkpoint = ModelCheckpoint(
            save_dir=ckpt_config['save_dir'],
            monitor=ckpt_config['monitor'],
            mode=ckpt_config['mode'],
            save_best_only=ckpt_config['save_best_only']
        )
        print(f"✅ Checkpointing: monitor={ckpt_config['monitor']}")
    
    def _setup_mlflow(self):
        """Configura MLflow para logging."""
        logging_config = self.config['logging']
        
        if logging_config.get('use_mlflow', True):
            mlflow.set_tracking_uri(logging_config['mlflow_uri'])
            mlflow.set_experiment(self.config['experiment']['name'])
            
            # Log de hiperparámetros
            mlflow.log_params({
                'model_type': self.config['model']['type'],
                'batch_size': self.config['training']['batch_size'],
                'learning_rate': self.config['training']['learning_rate'],
                'num_epochs': self.config['training']['num_epochs'],
                'optimizer': self.config['training']['optimizer'],
                'loss': self.config['training']['loss'],
                'seed': self.config['experiment']['seed']
            })
            
            print(f"✅ MLflow tracking URI: {logging_config['mlflow_uri']}")
    
    def train_epoch(self, epoch: int):
        """Entrena una época."""
        self.model.train()
        tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (image, clinical, weeks, target) in enumerate(pbar):
            # Mover a device
            image = image.to(self.device)
            clinical = clinical.to(self.device)
            weeks = weeks.to(self.device)
            target = target.to(self.device)
            
            # Forward
            if isinstance(self.model, MultimodalFusionModel):
                predictions = self.model(image, clinical, weeks)
            else:  # Clinical only
                predictions = self.model(clinical, weeks)
            
            # Loss - asegurar que target tenga el mismo shape que predictions
            target_flat = target.view(-1)  # Aplanar target
            loss = self.criterion(predictions, target_flat)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Actualizar métricas
            tracker.update(predictions, target_flat, loss.item())
            
            # Actualizar progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calcular métricas de la época
        metrics = tracker.compute()
        
        return metrics
    
    def validate(self):
        """Valida el modelo."""
        self.model.eval()
        tracker = MetricsTracker()
        
        with torch.no_grad():
            for image, clinical, weeks, target in tqdm(self.val_loader, desc="Validating"):
                image = image.to(self.device)
                clinical = clinical.to(self.device)
                weeks = weeks.to(self.device)
                target = target.to(self.device)
                
                # Forward
                if isinstance(self.model, MultimodalFusionModel):
                    predictions = self.model(image, clinical, weeks)
                else:
                    predictions = self.model(clinical, weeks)
                
                # Loss
                loss = self.criterion(predictions, target.squeeze())
                
                # Actualizar métricas
                tracker.update(predictions, target.squeeze(), loss.item())
        
        metrics = tracker.compute()
        
        return metrics
    
    def train(self):
        """Loop principal de entrenamiento."""
        print("\n" + "="*60)
        print("🚀 INICIANDO ENTRENAMIENTO")
        print("="*60 + "\n")
        
        num_epochs = self.config['training']['num_epochs']
        best_val_metric = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n📅 Época {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Entrenar
            train_metrics = self.train_epoch(epoch)
            print_metrics(train_metrics, prefix="Train")
            
            # Validar
            val_metrics = self.validate()
            print_metrics(val_metrics, prefix="Validation")
            
            # Log a MLflow
            if self.config['logging'].get('use_mlflow', True):
                for metric_name, value in train_metrics.items():
                    mlflow.log_metric(f"train_{metric_name}", value, step=epoch)
                for metric_name, value in val_metrics.items():
                    mlflow.log_metric(f"val_{metric_name}", value, step=epoch)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Checkpoint
            self.checkpoint(self.model, self.optimizer, epoch, val_metrics)
            
            # Early stopping
            if self.early_stopping is not None:
                monitor_metric = val_metrics[self.config['checkpoint']['monitor'].replace('val_', '')]
                if self.early_stopping(monitor_metric):
                    print(f"\n🛑 Early stopping en época {epoch}")
                    break
        
        print("\n" + "="*60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*60)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Entrenar modelo OSIC FPI')
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/configs/config.yaml',
        help='Ruta al archivo de configuración'
    )
    args = parser.parse_args()
    
    # Cargar configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("🏥 OSIC PULMONARY FIBROSIS PROGRESSION")
    print("="*60)
    print(f"📄 Config: {args.config}")
    print(f"🧪 Experimento: {config['experiment']['name']}")
    
    # Crear trainer y entrenar
    with mlflow.start_run():
        trainer = Trainer(config)
        trainer.train()
        
        # Log del modelo final
        mlflow.pytorch.log_model(trainer.model, "model")
    
    print("\n🎉 ¡Entrenamiento finalizado exitosamente!")


if __name__ == "__main__":
    main()