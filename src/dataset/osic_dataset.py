"""
Dataset PyTorch para datos multimodales OSIC:
- Imágenes CT (DICOM)
- Datos clínicos tabulares
- Target: FVC (Forced Vital Capacity)
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple

# Añadir src al path para imports
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.dicom_loader import DICOMLoader
from preprocessing.preprocessing import CTPreprocessor


class OSICDataset(Dataset):
    """
    Dataset multimodal para predicción de progresión de FPI.
    Combina imágenes CT y datos clínicos.
    """
    
    def __init__(
        self,
        csv_path: str,
        dicom_root: str,
        preprocessor: Optional[CTPreprocessor] = None,
        use_cache: bool = True,
        cache_dir: str = "data/processed/cache"
    ):
        """
        Args:
            csv_path: Ruta a train.csv o test.csv
            dicom_root: Carpeta con subdirectorios de pacientes DICOM
            preprocessor: Instancia de CTPreprocessor (si None, usa default)
            use_cache: Si cachear volúmenes procesados en disco
            cache_dir: Directorio para cache
        """
        self.df = pd.read_csv(csv_path)
        self.dicom_root = Path(dicom_root)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Preprocesador
        self.preprocessor = preprocessor or CTPreprocessor()
        
        # Loader DICOM
        self.dicom_loader = DICOMLoader(dicom_root)
        
        # Preparar datos clínicos
        self._prepare_clinical_data()
        
        # Filtrar registros sin FVC (target)
        self.df = self.df.dropna(subset=['FVC']).reset_index(drop=True)
        
        print(f"📊 Dataset cargado: {len(self.df)} registros de {self.df['Patient'].nunique()} pacientes")
    
    def _prepare_clinical_data(self):
        """Prepara y normaliza datos clínicos."""
        # One-hot encoding para Sex
        self.df['Sex_Male'] = (self.df['Sex'] == 'Male').astype(float)
        
        # One-hot para SmokingStatus
        smoking_dummies = pd.get_dummies(self.df['SmokingStatus'], prefix='Smoking')
        self.df = pd.concat([self.df, smoking_dummies], axis=1)
        
        # Normalizar variables numéricas
        self.df['Age_norm'] = self.df['Age'] / 100.0
        self.df['Weeks_norm'] = self.df['Weeks'] / 100.0
        self.df['FVC_norm'] = self.df['FVC'] / 5000.0
        self.df['Percent_norm'] = self.df['Percent'] / 100.0
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _get_cache_path(self, patient_id: str) -> Path:
        """Ruta de cache para volumen procesado."""
        return self.cache_dir / f"{patient_id}.npy"
    
    def _load_volume(self, patient_id: str) -> Optional[np.ndarray]:
        """
        Carga volumen (desde cache o procesando DICOM).
        Returns None si no se puede cargar.
        """
        cache_path = self._get_cache_path(patient_id)
        
        # Intentar cargar desde cache
        if self.use_cache and cache_path.exists():
            try:
                volume = np.load(cache_path)
                return volume
            except Exception as e:
                print(f"⚠️ Error cargando cache {cache_path}: {e}")
        
        # Cargar DICOM y procesar
        try:
            volume_hu, metadata = self.dicom_loader.load_patient_volume(patient_id)
            
            # Preparar spacing
            spacing = (
                metadata['slice_thickness'],
                metadata['pixel_spacing'][0],
                metadata['pixel_spacing'][1]
            )
            
            # Preprocesar
            volume = self.preprocessor.process(volume_hu, spacing, normalize=True)
            
            # Guardar en cache
            if self.use_cache:
                np.save(cache_path, volume)
            
            return volume
            
        except Exception as e:
            print(f"❌ Error procesando paciente {patient_id}: {e}")
            # Retornar None para indicar fallo
            return None
    
    def _get_clinical_features(self, row: pd.Series) -> np.ndarray:
        """
        Extrae vector de features clínicos.
        """
        features = [
            row['Age_norm'],
            row['Sex_Male'],
            row['Weeks_norm'],
            row['FVC_norm'],
            row['Percent_norm']
        ]
        
        # Añadir smoking status (one-hot)
        smoking_cols = [col for col in self.df.columns if col.startswith('Smoking_')]
        for col in smoking_cols:
            features.append(row.get(col, 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: Tensor (1, D, H, W) - volumen CT
            clinical: Tensor (n_features,) - datos clínicos
            weeks: Tensor (1,) - semanas de seguimiento
            target: Tensor (1,) - FVC objetivo
        """
        row = self.df.iloc[idx]
        
        # Cargar volumen
        volume = self._load_volume(row['Patient'])
        
        # Si el volumen falló, intentar con el siguiente índice
        max_retries = 10
        retry_count = 0
        while volume is None and retry_count < max_retries:
            print(f"⚠️ Saltando paciente {row['Patient']}, intentando siguiente...")
            idx = (idx + 1) % len(self)
            row = self.df.iloc[idx]
            volume = self._load_volume(row['Patient'])
            retry_count += 1
        
        # Si después de reintentos sigue fallando, usar volumen de ceros
        if volume is None:
            print(f"❌ No se pudo cargar ningún volumen después de {max_retries} intentos, usando volumen vacío")
            volume = np.zeros(self.preprocessor.target_size, dtype=np.float32)
        
        image = torch.from_numpy(volume).unsqueeze(0).float()  # (1, D, H, W)
        
        # Features clínicos
        clinical = torch.from_numpy(self._get_clinical_features(row)).float()
        
        # Weeks
        weeks = torch.tensor([row['Weeks_norm']], dtype=torch.float32)
        
        # Target - asegurar shape consistente
        target = torch.tensor(row['FVC'], dtype=torch.float32)
        
        return image, clinical, weeks, target


def test_dataset():
    """Test del dataset."""
    from torch.utils.data import DataLoader
    
    dataset = OSICDataset(
        csv_path="../../data/raw/train.csv",
        dicom_root="../../data/raw/train",
        use_cache=True
    )
    
    print(f"\n📦 Tamaño del dataset: {len(dataset)}")
    
    # Probar __getitem__
    image, clinical, weeks, target = dataset[0]
    print(f"\n🖼️ Image shape: {image.shape}")
    print(f"🩺 Clinical features shape: {clinical.shape}")
    print(f"📅 Weeks: {weeks.item():.3f}")
    print(f"🎯 Target FVC: {target.item():.1f}")
    
    # Probar DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    
    print(f"\n📦 Batch shapes:")
    print(f"  Images: {batch[0].shape}")
    print(f"  Clinical: {batch[1].shape}")
    print(f"  Weeks: {batch[2].shape}")
    print(f"  Targets: {batch[3].shape}")


if __name__ == "__main__":
    test_dataset()