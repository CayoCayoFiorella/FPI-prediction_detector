# FPI Prediction Detector

Este proyecto entrena un modelo multimodal para la predicción de progresión de fibrosis pulmonar idiopática (FPI).

## Tecnologías
- Python 3.x
- PyTorch
- MLflow
- Pylibjpeg / Pydicom

### Generar `requirements.txt`
Para guardar las dependencias actuales del entorno virtual:

```bash
pip freeze > requirements.txt
```

## Ejecución
```bash
# Limpiar cache
Remove-Item -Recurse -Force data\processed\cache\*

# Entrenar
python src/training/train.py --config experiments/configs/config.yaml
```
## Dataset
https://www.kaggle.com/competitions/osic-pulmonary-fibrosis-progression/data

