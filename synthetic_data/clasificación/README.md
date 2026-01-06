# Data Augmentation para Clasificación de Transposones

Este directorio contiene herramientas para generar datos sintéticos de transposones (TEs) con el objetivo de equilibrar clases minoritarias y mejorar la clasificación.

## Estructura y Uso

### 1. GAN (Generative Adversarial Networks)
Generación de secuencias sintéticas mediante modelos de Deep Learning (cGANs).

*   **Entrenamiento (`GAN/`)**:
    *   `run_gan_train.py`: Script principal para entrenar la cGAN (WGAN-GP o LSGAN).
    *   `train.sh`: Script de ejemplo para lanzar el entrenamiento en cluster.
    *   `utils/`: Definición de modelos (`models.py`), lógica de GANs (`gans.py`) y utilidades.

*   **Generación (`GAN/`)**:
    *   `run_generate_synth_data.py`: Genera datos sintéticos para equilibrar clases minoritarias usando el modelo entrenado.
    *   `datasynth.sh`: Script de ejemplo para la generación.

### 2. Data Augmentation Tradicional
Métodos clásicos de aumentación de datos.

*   `data_aug.py`: Script para generar variantes de secuencias existentes (mutaciones, ruido).
*   `data_aug.sh`: Script de ejecución.

