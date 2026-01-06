# Curación de Datos Sintéticos

Este directorio contiene herramientas para la generación de datos sintéticos y modelos de curación con clasificación tripartita (Kept, Removed, No_TE).

## Estructura

### 1. Models
Contiene los modelos de Deep Learning adaptados para la tarea de curación.
*   **Inpactor2**: Modelo basado en CNNs y k-mers.
    *   `inpactor2_model.py`: Script del modelo.
    *   `run.sh`: Ejemplo de ejecución.
*   **SENMAP**: Modelo basado en redes neuronales profundas.
    *   `senmap_model.py` / `train_SENMAP.py`: Scripts del modelo y entrenamiento.
    *   `run.sh`: Ejemplo de ejecución.

### 2. datasynth
Scripts para generar datasets sintéticos que simulan artefactos de ensamblaje y secuencias contaminantes.
*   `main.py`: Script principal de generación.
*   `run.sh`: Script de ejemplo para lanzar la generación en un cluster (SLURM).
*   `utils/`: Funciones auxiliares para la descarga y manipulación de secuencias.