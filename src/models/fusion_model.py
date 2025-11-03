"""
Modelo de fusión multimodal que combina:
- Características visuales (CNN 3D sobre imágenes CT)
- Características clínicas (MLP sobre datos tabulares)
Para predecir FVC (Forced Vital Capacity)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from models.cnn3d import CNN3D, CNN3DLightweight


class ClinicalEncoder(nn.Module):
    """Encoder MLP para datos clínicos tabulares."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        """
        Args:
            input_dim: Número de features clínicos de entrada
            hidden_dim: Dimensión de capa oculta
            output_dim: Dimensión de embedding clínico
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor (B, input_dim)
        Returns:
            Tensor (B, output_dim)
        """
        return self.network(x)


class FusionHead(nn.Module):
    """Cabezal de fusión que combina features y predice FVC."""
    
    def __init__(self, total_dim: int, hidden_dims: list = [256, 128]):
        """
        Args:
            total_dim: Dimensión total después de concatenar features
            hidden_dims: Lista de dimensiones para capas ocultas
        """
        super().__init__()
        
        layers = []
        in_dim = total_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            in_dim = hidden_dim
        
        # Capa final de predicción (sin activación, regresión)
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Tensor (B, total_dim)
        Returns:
            Tensor (B, 1) - predicción de FVC
        """
        return self.network(x)


class MultimodalFusionModel(nn.Module):
    """
    Modelo completo de fusión multimodal.
    Combina imágenes CT y datos clínicos para predecir progresión de FPI.
    """
    
    def __init__(
        self,
        clinical_input_dim: int,
        img_feature_dim: int = 128,
        clinical_embedding_dim: int = 32,
        fusion_hidden_dims: list = [256, 128],
        use_lightweight_cnn: bool = False
    ):
        """
        Args:
            clinical_input_dim: Número de features clínicos
            img_feature_dim: Dimensión de features de imagen
            clinical_embedding_dim: Dimensión de embedding clínico
            fusion_hidden_dims: Capas ocultas del fusion head
            use_lightweight_cnn: Si usar versión ligera de CNN (para CPU)
        """
        super().__init__()
        
        # CNN para imágenes
        if use_lightweight_cnn:
            self.image_encoder = CNN3DLightweight(
                in_channels=1,
                out_features=img_feature_dim
            )
        else:
            self.image_encoder = CNN3D(
                in_channels=1,
                out_features=img_feature_dim
            )
        
        # MLP para datos clínicos
        self.clinical_encoder = ClinicalEncoder(
            input_dim=clinical_input_dim,
            output_dim=clinical_embedding_dim
        )
        
        # Dimensión total después de concatenar
        # img_features + clinical_features + weeks
        total_dim = img_feature_dim + clinical_embedding_dim + 1
        
        # Fusion head
        self.fusion_head = FusionHead(
            total_dim=total_dim,
            hidden_dims=fusion_hidden_dims
        )
    
    def forward(self, image, clinical, weeks):
        """
        Args:
            image: Tensor (B, 1, D, H, W) - volumen CT
            clinical: Tensor (B, clinical_dim) - features clínicos
            weeks: Tensor (B, 1) - semanas de seguimiento
        
        Returns:
            prediction: Tensor (B, 1) - FVC predicho
        """
        # Extraer features de imagen
        img_features = self.image_encoder(image)  # (B, img_feature_dim)
        
        # Extraer features clínicos
        clin_features = self.clinical_encoder(clinical)  # (B, clinical_embedding_dim)
        
        # Concatenar todas las features
        fused = torch.cat([img_features, clin_features, weeks], dim=1)  # (B, total_dim)
        
        # Predicción
        prediction = self.fusion_head(fused)  # (B, 1)
        
        return prediction.squeeze(1)  # (B,)


class ClinicalOnlyModel(nn.Module):
    """Modelo baseline que solo usa datos clínicos (sin imágenes)."""
    
    def __init__(self, clinical_input_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(clinical_input_dim + 1, 128),  # +1 por weeks
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, clinical, weeks):
        x = torch.cat([clinical, weeks], dim=1)
        return self.network(x).squeeze(1)


def test_fusion_model():
    """Test del modelo de fusión."""
    print("🧪 Testing MultimodalFusionModel...\n")
    
    # Parámetros
    batch_size = 2
    clinical_dim = 8  # Ejemplo: age, sex, smoking (one-hot), etc.
    
    # Crear modelo
    model = MultimodalFusionModel(
        clinical_input_dim=clinical_dim,
        img_feature_dim=128,
        clinical_embedding_dim=32,
        use_lightweight_cnn=True  # Usar versión ligera para test
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parámetros totales: {total_params:,}\n")
    
    # Crear inputs dummy
    dummy_image = torch.randn(batch_size, 1, 64, 256, 256)
    dummy_clinical = torch.randn(batch_size, clinical_dim)
    dummy_weeks = torch.randn(batch_size, 1)
    
    print(f"📥 Input shapes:")
    print(f"  Image: {dummy_image.shape}")
    print(f"  Clinical: {dummy_clinical.shape}")
    print(f"  Weeks: {dummy_weeks.shape}\n")
    
    # Forward pass
    with torch.no_grad():
        prediction = model(dummy_image, dummy_clinical, dummy_weeks)
    
    print(f"📤 Output shape: {prediction.shape}")
    print(f"📤 Predictions: {prediction}\n")
    print(f"✅ Test exitoso!\n")
    
    # Test Clinical-only model
    print("🧪 Testing ClinicalOnlyModel...\n")
    clinical_model = ClinicalOnlyModel(clinical_input_dim=clinical_dim)
    clinical_params = sum(p.numel() for p in clinical_model.parameters())
    print(f"📊 Parámetros (clinical-only): {clinical_params:,}\n")
    
    with torch.no_grad():
        clinical_pred = clinical_model(dummy_clinical, dummy_weeks)
    
    print(f"📤 Output shape: {clinical_pred.shape}")
    print(f"📤 Predictions: {clinical_pred}\n")
    print(f"✅ Clinical model test exitoso!")


if __name__ == "__main__":
    test_fusion_model()