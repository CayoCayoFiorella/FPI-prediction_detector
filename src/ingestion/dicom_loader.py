"""
Módulo para cargar y procesar imágenes DICOM de tomografías pulmonares.
Convierte series DICOM a volúmenes 3D en unidades Hounsfield (HU).
Versión robusta que maneja diferentes formatos DICOM.
"""

import os
import numpy as np
import pydicom
from typing import Tuple, List, Optional
from pathlib import Path


class DICOMLoader:
    """Carga y procesa series DICOM de pacientes."""
    
    def __init__(self, dicom_root: str):
        """
        Args:
            dicom_root: Ruta raíz con carpetas de pacientes
        """
        self.dicom_root = Path(dicom_root)
        
    def load_patient_volume(self, patient_id: str) -> Tuple[np.ndarray, dict]:
        """
        Carga todos los cortes DICOM de un paciente y reconstruye el volumen 3D.
        
        Args:
            patient_id: ID del paciente
            
        Returns:
            volume: Array 3D (depth, height, width) en unidades Hounsfield
            metadata: Diccionario con información del volumen
        """
        patient_dir = self.dicom_root / patient_id
        
        if not patient_dir.exists():
            raise FileNotFoundError(f"No se encuentra carpeta: {patient_dir}")
        
        # Obtener archivos DICOM
        dcm_files = sorted(list(patient_dir.glob("*.dcm")))
        
        if len(dcm_files) == 0:
            raise ValueError(f"No hay archivos DICOM en {patient_dir}")
        
        # Leer slices y ordenar por posición
        slices = []
        for dcm_file in dcm_files:
            try:
                ds = pydicom.dcmread(str(dcm_file), force=True)
                slices.append(ds)
            except Exception as e:
                print(f"⚠️ Error leyendo {dcm_file}: {e}")
                continue
        
        if len(slices) == 0:
            raise ValueError(f"No se pudieron leer archivos DICOM en {patient_dir}")
        
        # Ordenar por ImagePositionPatient (Z)
        slices = self._sort_slices(slices)
        
        # Extraer metadata del primer slice
        first_slice = slices[0]
        
        # Obtener ImagePositionPatient de forma segura
        try:
            image_position = [float(x) for x in first_slice.ImagePositionPatient]
        except (AttributeError, TypeError):
            image_position = [0.0, 0.0, 0.0]  # Valor por defecto
        
        # Obtener PixelSpacing de forma segura
        try:
            pixel_spacing = [float(x) for x in first_slice.PixelSpacing]
        except (AttributeError, TypeError):
            pixel_spacing = [1.0, 1.0]  # Valor por defecto
        
        metadata = {
            'patient_id': patient_id,
            'num_slices': len(slices),
            'slice_thickness': float(getattr(first_slice, 'SliceThickness', 1.0)),
            'pixel_spacing': pixel_spacing,
            'image_position': image_position,
            'rows': int(getattr(first_slice, 'Rows', 512)),
            'columns': int(getattr(first_slice, 'Columns', 512))
        }
        
        # Construir volumen 3D
        volume = self._build_volume(slices)
        
        # Convertir a unidades Hounsfield
        volume_hu = self._convert_to_hu(volume, slices)
        
        return volume_hu, metadata
    
    def _sort_slices(self, slices: List) -> List:
        """Ordena slices por posición Z de forma robusta."""
        try:
            # Intentar ordenar por ImagePositionPatient
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except (AttributeError, TypeError):
            try:
                # Si no tiene ImagePositionPatient, usar InstanceNumber
                slices.sort(key=lambda x: int(x.InstanceNumber))
            except (AttributeError, TypeError):
                # Si tampoco tiene InstanceNumber, usar SliceLocation
                try:
                    slices.sort(key=lambda x: float(x.SliceLocation))
                except (AttributeError, TypeError):
                    # Último recurso: mantener orden de archivos
                    print("⚠️ No se pudo ordenar slices, usando orden de archivos")
        return slices
    
    def _build_volume(self, slices: List) -> np.ndarray:
        """Construye array 3D desde lista de slices."""
        slice_arrays = []
        for s in slices:
            try:
                slice_arrays.append(s.pixel_array)
            except Exception as e:
                print(f"⚠️ Error extrayendo pixel_array: {e}")
                continue
        
        if len(slice_arrays) == 0:
            raise ValueError("No se pudieron extraer imágenes de los slices")
        
        volume = np.stack(slice_arrays, axis=0)
        return volume
    
    def _convert_to_hu(self, volume: np.ndarray, slices: List) -> np.ndarray:
        """
        Convierte valores de pixel a unidades Hounsfield (HU).
        HU = pixel_value * RescaleSlope + RescaleIntercept
        """
        first_slice = slices[0]
        intercept = float(getattr(first_slice, 'RescaleIntercept', 0))
        slope = float(getattr(first_slice, 'RescaleSlope', 1))
        
        volume_hu = volume.astype(np.float32)
        volume_hu = volume_hu * slope + intercept
        
        return volume_hu


def test_loader():
    """Función de prueba rápida."""
    loader = DICOMLoader("../../data/raw/train")
    
    # Obtener primer paciente
    import pandas as pd
    df = pd.read_csv("../../data/raw/train.csv")
    first_patient = df['Patient'].iloc[0]
    
    print(f"🔍 Cargando paciente: {first_patient}")
    
    try:
        volume, metadata = loader.load_patient_volume(first_patient)
        print(f"✅ Volumen cargado: {volume.shape}")
        print(f"📊 Rango HU: [{volume.min():.1f}, {volume.max():.1f}]")
        print(f"📋 Metadata: {metadata}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_loader()