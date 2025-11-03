"""
Script para validar archivos DICOM antes del entrenamiento.
Detecta archivos corruptos o no decodificables en cada paciente.
"""

import os
import csv
import pydicom
from pathlib import Path
from tqdm import tqdm

def check_dicoms(root_dir: str, output_csv: str = "dicom_validation_report.csv"):
    """
    Valida todos los archivos DICOM en el dataset.
    
    Args:
        root_dir: Carpeta raíz que contiene las carpetas de pacientes (cada una con sus DICOMs)
        output_csv: Nombre del archivo CSV donde guardar el reporte
    """
    root = Path(root_dir)
    patients = sorted([p for p in root.iterdir() if p.is_dir()])

    if not patients:
        print(f"❌ No se encontraron carpetas de pacientes en: {root}")
        return

    results = []

    print("============================================================")
    print("🔍 VALIDANDO ARCHIVOS DICOM")
    print("============================================================")

    for patient in tqdm(patients, desc="Revisando pacientes", ncols=100):
        dcm_files = list(patient.glob("*.dcm"))
        total = len(dcm_files)
        bad_files = 0
        bad_list = []

        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f, force=True)
                _ = ds.pixel_array  # fuerza la decodificación
            except Exception as e:
                bad_files += 1
                bad_list.append(f.name)

        results.append({
            "patient_id": patient.name,
            "total_dicoms": total,
            "corrupt_dicoms": bad_files,
            "corrupt_percentage": round(bad_files / total * 100, 2) if total > 0 else 0.0,
            "corrupt_files": "; ".join(bad_list[:5]) + (" ..." if len(bad_list) > 5 else "")
        })

    # Guardar CSV
    output_path = Path(output_csv)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n============================================================")
    print(f"✅ Validación completada. Reporte guardado en: {output_path}")
    print("============================================================")

    # Resumen global
    total_patients = len(results)
    corrupt_patients = sum(1 for r in results if r["corrupt_dicoms"] > 0)
    total_corrupt = sum(r["corrupt_dicoms"] for r in results)
    total_files = sum(r["total_dicoms"] for r in results)

    print(f"🧾 Pacientes totales: {total_patients}")
    print(f"⚠️ Pacientes con DICOMs corruptos: {corrupt_patients}")
    print(f"🩻 Total DICOMs revisados: {total_files}")
    print(f"💀 Total DICOMs corruptos: {total_corrupt}")
    print(f"📊 Porcentaje global de errores: {round(total_corrupt / max(total_files, 1) * 100, 2)}%")

    print("============================================================")
    print("🔹 Revisa el CSV para ver los pacientes afectados.")
    print("============================================================")


if __name__ == "__main__":
    # Ruta raíz de tus DICOMs — ajústala si es distinta
    ROOT_DIR = "data/raw/train"
    OUTPUT_CSV = "dicom_validation_report.csv"
    
    check_dicoms(ROOT_DIR, OUTPUT_CSV)
