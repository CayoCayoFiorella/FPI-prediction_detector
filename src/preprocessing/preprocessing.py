"""
Módulo de preprocesamiento para volúmenes de CT pulmonar.
Incluye: windowing, resampling, normalización y cropping.
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from typing import Tuple, Optional


class CTPreprocessor:
    """Preprocesador para volúmenes de tomografía computarizada."""
    
    def __init__(
        self,
        target_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
        target_size: Tuple[int, int, int] = (64, 256, 256),
        lung_window: Tuple[float, float] = (-1000, 400),
        clip_range: Tuple[float, float] = (-1000, 400)
    ):
        """
        Args:
            target_spacing: Espaciado isotrópico objetivo (Z, Y, X) en mm
            target_size: Tamaño final del volumen (D, H, W)
            lung_window: Ventana pulmonar (min_HU, max_HU)
            clip_range: Rango para recortar valores extremos
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.lung_window = lung_window
        self.clip_range = clip_range
    
    def process(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Pipeline completo de preprocesamiento.
        
        Args:
            volume: Volumen 3D en unidades Hounsfield
            original_spacing: Espaciado original (slice_thickness, pixel_spacing[0], pixel_spacing[1])
            normalize: Si normalizar a rango [0, 1]
            
        Returns:
            Volumen procesado
        """
        # 1. Aplicar ventana pulmonar
        volume = self.apply_lung_window(volume)
        
        # 2. Resamplear a espaciado isotrópico
        volume = self.resample(volume, original_spacing)
        
        # 3. Segmentación simple de pulmones (opcional, mejora resultados)
        # mask = self.simple_lung_mask(volume)
        # volume = volume * mask
        
        # 4. Crop/Pad a tamaño objetivo
        volume = self.crop_or_pad(volume, self.target_size)
        
        # 5. Normalizar
        if normalize:
            volume = self.normalize(volume)
        
        return volume
    
    def apply_lung_window(self, volume: np.ndarray) -> np.ndarray:
        """Aplica ventana pulmonar y clip de valores."""
        volume = np.clip(volume, self.clip_range[0], self.clip_range[1])
        return volume
    
    def resample(
        self,
        volume: np.ndarray,
        original_spacing: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Resamplea volumen a espaciado isotrópico usando interpolación.
        """
        original_spacing = np.array(original_spacing)
        target_spacing = np.array(self.target_spacing)
        
        # Calcular factor de zoom
        zoom_factors = original_spacing / target_spacing
        
        # Resamplear
        resampled = ndimage.zoom(
            volume,
            zoom_factors,
            order=1,  # Interpolación bilineal
            mode='constant',
            cval=-1000  # Valor de fondo (aire)
        )
        
        return resampled
    
    def simple_lung_mask(self, volume: np.ndarray, threshold: float = -400) -> np.ndarray:
        """
        Crea máscara simple de pulmones usando umbralización.
        Nota: Para producción, usar modelos pre-entrenados (U-Net).
        """
        # Umbralizar: tejido pulmonar es < -400 HU aprox
        mask = volume < threshold
        
        # Limpiar con operaciones morfológicas
        from scipy.ndimage import binary_opening, binary_closing
        mask = binary_closing(mask, iterations=2)
        mask = binary_opening(mask, iterations=2)
        
        return mask.astype(np.float32)
    
    def crop_or_pad(
        self,
        volume: np.ndarray,
        target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Ajusta volumen al tamaño objetivo mediante crop central o padding.
        """
        current_size = np.array(volume.shape)
        target_size = np.array(target_size)
        
        # Calcular padding/crop necesario
        diff = target_size - current_size
        
        # Crear volumen de salida
        output = np.zeros(target_size, dtype=volume.dtype) - 1000  # Fondo = aire
        
        # Calcular índices de inicio para colocación central
        start_current = np.maximum(-diff // 2, 0)
        start_target = np.maximum(diff // 2, 0)
        
        # Calcular tamaños de copia
        copy_size = np.minimum(current_size - start_current, target_size - start_target)
        
        # Copiar datos
        output[
            start_target[0]:start_target[0] + copy_size[0],
            start_target[1]:start_target[1] + copy_size[1],
            start_target[2]:start_target[2] + copy_size[2]
        ] = volume[
            start_current[0]:start_current[0] + copy_size[0],
            start_current[1]:start_current[1] + copy_size[1],
            start_current[2]:start_current[2] + copy_size[2]
        ]
        
        return output
    
    def normalize(self, volume: np.ndarray) -> np.ndarray:
        """Normaliza volumen a rango [0, 1] basado en lung_window."""
        min_hu, max_hu = self.lung_window
        volume = (volume - min_hu) / (max_hu - min_hu)
        volume = np.clip(volume, 0, 1)
        return volume.astype(np.float32)


def test_preprocessor():
    """Prueba del preprocesador."""
    # Crear volumen dummy
    dummy_volume = np.random.randn(100, 512, 512) * 200 - 500  # HU simulado
    original_spacing = (2.5, 0.7, 0.7)  # spacing 
    
    preprocessor = CTPreprocessor(
        target_spacing=(1.5, 1.5, 1.5),
        target_size=(64, 256, 256)
    )
    
    print(f"📥 Volumen original: {dummy_volume.shape}, rango: [{dummy_volume.min():.1f}, {dummy_volume.max():.1f}]")
    
    processed = preprocessor.process(dummy_volume, original_spacing)
    
    print(f"📤 Volumen procesado: {processed.shape}, rango: [{processed.min():.3f}, {processed.max():.3f}]")
    print(f"✅ Preprocesamiento exitoso!")


if __name__ == "__main__":
    test_preprocessor()