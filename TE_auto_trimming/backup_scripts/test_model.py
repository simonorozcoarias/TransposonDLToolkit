import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score as r2_sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import argparse, re

# === Clase NDStandardScaler ===
class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X

    def save_model(self, model_name):
        dump(self._scaler, model_name + '.bin')

    def load_model(self, model_path, X):
        self._scaler = load(model_path)
        self._orig_shape = X.shape[1:]

# === Funcion r2_score para Keras ===
def r2_score(y_true, y_pred):
    sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_y_true = tf.reduce_mean(y_true)
    sum_squares = tf.reduce_sum(tf.square(y_true - mean_y_true))

    epsilon = tf.keras.backend.epsilon()
    sum_squares = tf.maximum(sum_squares, epsilon)

    return 1 - sum_squares_residuals / sum_squares

# === Argumentos con argparse ===
parser = argparse.ArgumentParser(description="SmartInspection Prediction Script")
parser.add_argument("model_path", type=str, help="Path to trained Keras model")
parser.add_argument("scalerX_path", type=str, help="Path to saved scaler")
parser.add_argument("data_path", type=str, help="Path to X_test .npy file")
args = parser.parse_args()

def calcular_y_graficar_r23(real, predicted, nombre):
    print(f"Calculating R2 for {nombre}...")
    print(f"NaNs in real: {np.isnan(real).sum()}, NaNs in predicted: {np.isnan(predicted).sum()}")
    print(f"real shape: {real.shape}, predicted shape: {predicted.shape}")

    # Verificar que las formas de los datos coincidan
    if real.shape != predicted.shape:
        raise ValueError("Shapes of real and predicted do not match.")

    # Calcular el coeficiente de determinacion R^2
    r2 = r2_score(real, predicted)

    # Crear el grafico
    plt.figure(figsize=(8, 6))
    plt.scatter(real, predicted, color='blue', label='Datos')
    plt.plot(real, real, color='red', label=f'Referencia (y=x, R^2 = {r2:.2f})')
    plt.xlabel('Real')
    plt.ylabel('Predicted')
    plt.title('Grafico de dispersion con R^2 - ' + nombre)
    plt.legend()
    plt.grid(True)
    plt.savefig('r2_' + nombre + '.png', bbox_inches='tight', dpi=500)

    return r2
    
def testing_models2(model_path, scalerx_path, data_path):

    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # --- Cargar datos ---
    X_test = np.load(os.path.join(data_path, "features_data.npy"))
    X_test = X_test / 255.0
    X_test = X_test.astype(np.float32)    
    
    Y_test = np.load(os.path.join(data_path, "labels_data.npy")).astype(np.float32)
    case_labels = np.load(os.path.join(data_path, "case_labels.npy"))
    
    sample_names = np.arange(X_test.shape[0])
    
    unique_cases = np.unique(case_labels)
    print(f"{unique_cases}")
    metrics = {}

    sample_names = np.arange(X_test.shape[0])
    output_dir = "./results_cases"
    os.makedirs(output_dir, exist_ok=True)
    
    print("X_test raw:", np.min(X_test), np.max(X_test))
    print("Y_test raw:", np.min(Y_test), np.max(Y_test))
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    print("NaNs in X_test:", np.isnan(X_test).sum())
    print("NaNs in Y_test:", np.isnan(Y_test).sum())
    
    # === Load scaler ===
    scalerX = NDStandardScaler()
    scalerX.load_model(scalerx_path, X_test)
    X1_test_scl = scalerX.transform(X_test)
    
    print("NaNs in X1_test_scl after scaling:", np.isnan(X1_test_scl).sum())

    # === Load model ===
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'LeakyReLU': tf.keras.layers.LeakyReLU(0.1),
            'r2_score': r2_score
        }
    )

    # === Predict ===
    predictions = model.predict(
        [
            X1_test_scl[:, :, :, 0],
            X1_test_scl[:, :, :, 1],
            X1_test_scl[:, :, :, 2],
            X1_test_scl[:, :, :, 3]
        ],
        verbose=0
    )
    predictions = np.nan_to_num(predictions, nan=0)
    
    print("NaNs in predictions:", np.isnan(predictions).sum())
    print("predictions shape:", predictions.shape)
    
    for case in unique_cases:
    
        case = re.sub(r'[\\/:"*?<>|#]', '_', case)
    
        idx = np.where(case_labels == case)[0]
        
        print(f"{idx}")
        
        if len(idx) == 0:
            print(f"Warning: No samples found for case '{case}', skipping...")
            continue
        
        y_true = Y_test[idx]
        print(f"{y_true}")
        y_pred = predictions[idx]
        print(f"{y_pred}")
    
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    
        metrics[case] = {"MSE": mse, "MAE": mae, "R2": r2}
        print(f"Case {case}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    
        # --- Scatter plot por caso ---
        plt.figure(figsize=(6,6))
    
        # Output 0
        plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6, color='blue', label='Output0')
        # Output 1
        plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.6, color='green', marker='x', label='Output1')
    
        # Linea de referencia y=x
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
    
        plt.title(f"Caso {case}")
        plt.xlabel("Real")
        plt.ylabel("Predicho")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_case_{case}.png"))
        plt.close()
    
        # --- Calcular R2 por posicion ---
        r2_initial = calcular_y_graficar_r23(Y_test[idx, 0], predictions[idx, 0], f"StartingPos_{case}")
        r2_final = calcular_y_graficar_r23(Y_test[idx, 1], predictions[idx, 1], f"EndingPos_{case}")
    
        print(f"R2 starting position_{case}: {r2_initial:.4f}")
        print(f"R2 ending position_{case}: {r2_final:.4f}")

    df = pd.DataFrame({
    "Sample": sample_names,
    **{f"Real_Output{i}": Y_test[:, i] for i in range(Y_test.shape[1])},
    **{f"Pred_Output{i}": predictions[:, i] for i in range(predictions.shape[1])}
    })
    df.to_csv(os.path.join(output_dir, "predicciones_metrics.csv"), index=False)

    # === Guardar resumen de metricas ===
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=True)

    print(f"\nPredicciones guardadas en '{os.path.join(output_dir, 'predicciones_metrics.csv')}'")
    print(f"Resumen de metricas guardado en '{os.path.join(output_dir, 'metrics_summary.csv')}'")

testing_models2(args.model_path, args.scalerX_path, args.data_path)
