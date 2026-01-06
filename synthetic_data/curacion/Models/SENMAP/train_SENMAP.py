#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import datetime
import math

# Import model definition
from senmap_model import get_senmap_model, f1_m

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# GPU Memory Growth Fix
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# mapeo:
# 1: Kept (casos reales, no empezando por Caso)
# 0: Removed (Caso 1, 2, 3)
# 2: No_TE (Caso 4, 5)
CASE_TO_CLASS = {
    'Caso1': 0,
    'Caso2': 0,
    'Caso3': 0,
    'Caso4': 2,
    'Caso5': 2
}

class SenmapDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, sequences, labels=None, batch_size=32, max_len=23200, shuffle=True, num_classes=3):
        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.indexes = np.arange(len(self.sequences))
        self.langu = ['A', 'C', 'G', 'T', 'N']
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.sequences) / self.batch_size)

    def __getitem__(self, index):
        # Genera indices del batch
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.sequences))
        batch_indexes = self.indexes[start_index:end_index]

        X = np.zeros((len(batch_indexes), 5, self.max_len, 1), dtype=np.float32)
        
        for i, idx in enumerate(batch_indexes):
            seq_record = self.sequences[idx]
            seq = str(seq_record.seq).upper()
            
            # Padding
            if len(seq) < self.max_len:
                pad_total = self.max_len - len(seq)
                pad_left = pad_total // 2
                seq = 'N' * pad_left + seq + 'N' * (pad_total - pad_left)
            else:
                seq = seq[:self.max_len]
                
            for j, nucl in enumerate(seq):
                if nucl in self.langu:
                    row = self.langu.index(nucl)
                    X[i, row, j, 0] = 1.0
                else:
                    X[i, 4, j, 0] = 1.0

        if self.labels is not None:
            y = self.labels[batch_indexes]
            return X, to_categorical(y, num_classes=self.num_classes)
        else:
            return X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

class SenmapModel:
    def __init__(self, model_path=None):
        self.input_shape = (5, 23200, 1)
        self.num_classes = 3
        
        if model_path and os.path.exists(model_path):
            print(f"Cargando modelo desde {model_path}...")
            self.model = tf.keras.models.load_model(model_path, custom_objects={'f1_m': f1_m})
        else:
            print(f"Creando nuevo modelo SENMAP con forma de entrada: {self.input_shape}")
            self.model = get_senmap_model(input_shape=self.input_shape, num_classes=self.num_classes)
            self.model.compile(optimizer='adam',
                               loss='categorical_crossentropy',
                               metrics=['accuracy', f1_m])

    def prepare_training_data(self, fasta_file):
        print(f"Cargando secuencias desde {fasta_file}...")
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        num_seqs = len(sequences)
        
        if num_seqs == 0:
            print(f"No se encontraron secuencias en {fasta_file}.")
            return None, None
            
        print(f"Se encontraron {num_seqs} secuencias.")
        
        Y = np.zeros(num_seqs, dtype=np.int32)
        
        for i, record in enumerate(sequences):
            seq_id = record.id
            # Real si no empieza con "Caso", Kept (1)
            if not seq_id.startswith("Caso"):
                Y[i] = 1
            else:
                # Casos sintéticos: Removed (0) o No_TE (2)
                found_case = False
                for case, class_idx in CASE_TO_CLASS.items():
                    if seq_id.startswith(case):
                        Y[i] = class_idx
                        found_case = True
                        break
                
                if not found_case:
                    print(f"Warning: Caso sintético no encontrado en el mapeo para {seq_id}. Defaulting to 0.")
                    Y[i] = 0
                    
        return sequences, Y

    def train(self, fasta_file, output_dir, epochs=50, batch_size=32):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        os.environ.pop('TMPDIR', None)

        sequences, Y = self.prepare_training_data(fasta_file)
        if sequences is None: return
        
        # split 70% train, 15% val, 15% test
        seq_temp, seq_test, Y_temp, Y_test = train_test_split(sequences, Y, test_size=0.15, random_state=42, stratify=Y)
        seq_train, seq_val, Y_train, Y_val = train_test_split(seq_temp, Y_temp, test_size=0.176, random_state=42, stratify=Y_temp)
        
        print(f"Training set: {len(seq_train)}")
        print(f"Validation set: {len(seq_val)}")
        print(f"Test set: {len(seq_test)}")
        
        # Generators
        train_gen = SenmapDataGenerator(seq_train, Y_train, batch_size=batch_size, shuffle=True)
        val_gen = SenmapDataGenerator(seq_val, Y_val, batch_size=batch_size, shuffle=False)
        test_gen = SenmapDataGenerator(seq_test, Y_test, batch_size=batch_size, shuffle=False)
        
        self.model.summary()
        
        # Callbacks
        checkpoint_path = os.path.join(output_dir, "best_model.h5")
        log_dir = os.path.join(output_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        callbacks = [
            ModelCheckpoint(checkpoint_path, monitor='val_f1_m', mode='max', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]
        
        print("\nStarting training...")
        self.model.fit(train_gen,
                       validation_data=val_gen,
                       epochs=epochs,
                       callbacks=callbacks)
        
        final_model_path = os.path.join(output_dir, "trained_model.h5")
        self.model.save(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        # Evaluar en test
        print("------------------PREDICT CON TEST ------------------")
        
        best_model = tf.keras.models.load_model(checkpoint_path, custom_objects={'f1_m': f1_m})

        Y_pred_probs = best_model.predict(test_gen)
        Y_pred = np.argmax(Y_pred_probs, axis=1)
        
        target_names = ['Removed (0)', 'Kept (1)', 'No_TE (2)']
        print(classification_report(Y_test, Y_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(Y_test, Y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Test Set)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_test.png'))
        print(f"Confusion matrix guardada en {os.path.join(output_dir, 'confusion_matrix_test.png')}")
        
        # Metricas
        print("\nGenerating detailed metrics per case...")
        stats = {
            'Caso1': {'total': 0, 'correct': 0, 'pred_0': 0, 'pred_1': 0, 'pred_2': 0},
            'Caso2': {'total': 0, 'correct': 0, 'pred_0': 0, 'pred_1': 0, 'pred_2': 0},
            'Caso3': {'total': 0, 'correct': 0, 'pred_0': 0, 'pred_1': 0, 'pred_2': 0},
            'Caso4': {'total': 0, 'correct': 0, 'pred_0': 0, 'pred_1': 0, 'pred_2': 0},
            'Caso5': {'total': 0, 'correct': 0, 'pred_0': 0, 'pred_1': 0, 'pred_2': 0},
            'Real':  {'total': 0, 'correct': 0, 'pred_0': 0, 'pred_1': 0, 'pred_2': 0}
        }
        
        for i, seq_record in enumerate(seq_test):
            seq_id = seq_record.id
            true_label = Y_test[i]
            pred_label = Y_pred[i]
            
            # determinar categoria
            category = 'Real'
            if seq_id.startswith("Caso"):
                for case in ['Caso1', 'Caso2', 'Caso3', 'Caso4', 'Caso5']:
                    if seq_id.startswith(case):
                        category = case
                        break
            
            # actualizar estadisticas
            if category in stats:
                stats[category]['total'] += 1
                if pred_label == true_label:
                    stats[category]['correct'] += 1
                    
                if pred_label == 0:
                    stats[category]['pred_0'] += 1
                elif pred_label == 1:
                    stats[category]['pred_1'] += 1
                elif pred_label == 2:
                    stats[category]['pred_2'] += 1
                
        # Report
        header = f"{'Case':<10} | {'Total':<8} | {'Accuracy':<10} | {'Pred Removed (0)':<16} | {'Pred Kept (1)':<13} | {'Pred No_TE (2)':<14}"
        separator = "-" * len(header)
        
        output_lines = []
        output_lines.append(header)
        output_lines.append(separator)
        
        for cat in ['Caso1', 'Caso2', 'Caso3', 'Caso4', 'Caso5', 'Real']:
            s = stats[cat]
            if s['total'] > 0:
                acc = (s['correct'] / s['total']) * 100
            else:
                acc = 0.0
                
            line = f"{cat:<10} | {s['total']:<8} | {acc:<8.2f} % | {s['pred_0']:<16} | {s['pred_1']:<13} | {s['pred_2']:<14}"
            output_lines.append(line)
            print(line)
            
        metrics_file = os.path.join(output_dir, 'detailed_metrics_by_case.txt')
        with open(metrics_file, 'w') as f:
            f.write("\n".join(output_lines))
        print(f"metrics guardadas en {metrics_file}")

    def predict(self, fasta_file, output_file=None):
        print(f"PREDICCIÓN")
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        if not sequences:
            print(f"No se han encontrado secuencias en {fasta_file}")
            return

        pred_gen = SenmapDataGenerator(sequences, labels=None, batch_size=32, shuffle=False)
        
        predictions = self.model.predict(pred_gen)
        predicted_classes = np.argmax(predictions, axis=1)
        
        results = []
        for i, seq_record in enumerate(sequences):
            pred_class = predicted_classes[i]
            confidence = predictions[i][pred_class]
            
            status = "Unknown"
            if pred_class == 0:
                status = "Removed (Needs curation)"
            elif pred_class == 1:
                status = "Kept (No curation needed)"
            elif pred_class == 2:
                status = "No TE"
                
            results.append((seq_record.id, pred_class, confidence, status))
            
        if output_file:
            with open(output_file, 'w') as f:
                f.write("ID\tClass\tConfidence\tStatus\n")
                for res in results:
                    f.write(f"{res[0]}\t{res[1]}\t{res[2]:.4f}\t{res[3]}\n")
            print(f"Results saved to {output_file}")
        else:
            for res in results:
                print(f"{res[0]}\t{res[1]}\t{res[2]:.4f}\t{res[3]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train/Predict SENMAP")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train")
    train_parser.add_argument('-i', '--input', required=True, help="fichero FASTA")
    train_parser.add_argument('-o', '--output', required=True, help="Directorio de salida")
    train_parser.add_argument('-e', '--epochs', type=int, default=500, help="Número de epochs")
    train_parser.add_argument('-b', '--batch_size', type=int, default=32, help="Tamaño del batch")
    
    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict")
    predict_parser.add_argument("-f", "--fasta", required=True, help="Fichero FASTA")
    predict_parser.add_argument("-m", "--model", required=True, help="Path al modelo entrenado")
    predict_parser.add_argument("-o", "--output", help="Fichero de salida")
    
    args = parser.parse_args()
    
    if args.command == "train":
        senmap = SenmapModel()
        senmap.train(args.input, args.output, args.epochs, args.batch_size)
        
    elif args.command == "predict":
        if not os.path.exists(args.fasta):
            print(f"Error: File {args.fasta} not found.")
            sys.exit(1)
        senmap = SenmapModel(args.model)
        senmap.predict(args.fasta, args.output)
        
    else:
        parser.print_help()
