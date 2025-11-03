"""
Red neuronal convolucional 3D para procesar volúmenes de CT pulmonar.
Extrae características visuales de las imágenes tomográficas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Bloque convolucional 3D con BatchNorm y ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CNN3D(nn.Module):
    """
    CNN 3D para extraer características de volúmenes CT.
    Arquitectura progresiva con downsampling.
    """
    
    def __init__(self, in_channels: int = 1, out_features: int = 128):
        """
        Args:
            in_channels: Número de canales de entrada (1 para CT grayscale)
            out_features: Dimensión del vector de características de salida
        """
        super().__init__()
        
        # Encoder: serie de bloques conv + pooling
        self.conv1 = ConvBlock3D(in_channels, 16)
        self.conv2 = ConvBlock3D(16, 32)
        
        self.conv3 = ConvBlock3D(32, 64)
        self.conv4 = ConvBlock3D(64, 64)
        
        self.conv5 = ConvBlock3D(64, 128)
        self.conv6 = ConvBlock3D(128, 128)
        
        # Pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Global pooling para reducir dimensiones espaciales
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Capa fully connected final
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_features)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor (B, 1, D, H, W)
        
        Returns:
            features: Tensor (B, out_features)
        """
        # Encoder pathway
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)  # Reduce a D/2, H/2, W/2
        
        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)  # Reduce a D/4, H/4, W/4
        
        # Block 3
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool(x)  # Reduce a D/8, H/8, W/8
        
        # Global pooling: (B, 128, D/8, H/8, W/8) -> (B, 128, 1, 1, 1)
        x = self.global_pool(x)
        
        # Flatten: (B, 128, 1, 1, 1) -> (B, 128)
        x = x.view(x.size(0), -1)
        
        # FC layers para obtener features
        features = self.fc(x)  # (B, out_features)
        
        return features


class CNN3DLightweight(nn.Module):
    """
    Versión ligera de CNN 3D para máquinas sin GPU potente.
    Menos capas y filtros.
    """
    
    def __init__(self, in_channels: int = 1, out_features: int = 128):
        super().__init__()
        
        self.conv1 = ConvBlock3D(in_channels, 8)
        self.conv2 = ConvBlock3D(8, 16)
        self.conv3 = ConvBlock3D(16, 32)
        self.conv4 = ConvBlock3D(32, 64)
        
        self.pool = nn.MaxPool3d(2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(64, out_features),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.global_pool(x)
        
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        return features


def test_cnn3d():
    """Test de la arquitectura CNN 3D."""
    print("🧪 Testing CNN3D...")
    
    # Crear modelo
    model = CNN3D(in_channels=1, out_features=128)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Parámetros totales: {total_params:,}")
    print(f"📊 Parámetros entrenables: {trainable_params:,}")
    
    # Input dummy (batch=2, channels=1, depth=64, height=256, width=256)
    dummy_input = torch.randn(2, 1, 64, 256, 256)
    print(f"\n📥 Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"📤 Output shape: {output.shape}")
    print(f"✅ Test exitoso!")
    
    # Test lightweight version
    print("\n🧪 Testing CNN3DLightweight...")
    model_light = CNN3DLightweight(out_features=128)
    light_params = sum(p.numel() for p in model_light.parameters())
    print(f"📊 Parámetros (lightweight): {light_params:,}")
    
    with torch.no_grad():
        output_light = model_light(dummy_input)
    print(f"📤 Output shape: {output_light.shape}")
    print(f"✅ Lightweight test exitoso!")
    
    return model


if __name__ == "__main__":
    test_cnn3d()
