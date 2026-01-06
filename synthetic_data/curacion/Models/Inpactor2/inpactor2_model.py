#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from joblib import load
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import datetime
import math

import warnings
warnings.filterwarnings('ignore')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def f1_m(y_true, y_pred):
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def fasta2one_hot(sequence, total_win_len):
    langu = ['A', 'C', 'G', 'T', 'N']
    posNucl = 0
    
    real_len = len(sequence)
    rep2d = np.zeros((1, 5, real_len), dtype=bool)

    for nucl in sequence:
        nucl_upper = nucl.upper()
        if nucl_upper in langu:
            posLang = langu.index(nucl_upper)
        else:
            posLang = langu.index('N')
        rep2d[0][posLang][posNucl] = 1
        posNucl += 1
    return rep2d

def kmer_extractor_model(input_shape):
    installation_path = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(installation_path, 'Models', 'Weights_SL.npy')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"pesos no encontrados en {weights_path}")
        
    weights = np.load(weights_path, allow_pickle=True)
    W_1, b_1 = weights[0], weights[1]
    W_2, b_2 = weights[2], weights[3]
    W_3, b_3 = weights[4], weights[5]
    W_4, b_4 = weights[6], weights[7]
    W_5, b_5 = weights[8], weights[9]
    W_6, b_6 = weights[10], weights[11]

    # modelo
    inputs = tf.keras.Input(shape=input_shape, name="input_1")
    
    layers_1 = tf.keras.layers.Conv2D(4, (5, 1), strides=(1, 1), weights=[W_1, b_1], activation='relu', use_bias=True, name='k_1')(inputs)
    layers_1 = tf.keras.backend.sum(layers_1, axis=-2)
    
    layers_2 = tf.keras.layers.Conv2D(16, (5, 2), strides=(1, 1), weights=[W_2, b_2], activation='relu', use_bias=True, name='k_2')(inputs)
    layers_2 = tf.keras.backend.sum(layers_2, axis=-2)

    layers_3 = tf.keras.layers.Conv2D(64, (5, 3), strides=(1, 1), weights=[W_3, b_3], activation='relu', use_bias=True, name='k_3')(inputs)
    layers_3 = tf.keras.backend.sum(layers_3, axis=-2)

    layers_4 = tf.keras.layers.Conv2D(256, (5, 4), strides=(1, 1), weights=[W_4, b_4], activation='relu', use_bias=True, name='k_4')(inputs)
    layers_4 = tf.keras.backend.sum(layers_4, axis=-2)

    layers_5 = tf.keras.layers.Conv2D(1024, (5, 5), strides=(1, 1), weights=[W_5, b_5], activation='relu', use_bias=True, name='k_5')(inputs)
    layers_5 = tf.keras.backend.sum(layers_5, axis=-2)

    layers_6 = tf.keras.layers.Conv2D(4096, (5, 6), strides=(1, 1), weights=[W_6, b_6], activation='relu', use_bias=True, name='k_6')(inputs)
    layers_6 = tf.keras.backend.sum(layers_6, axis=-2)

    layers = tf.concat([layers_1, layers_2, layers_3, layers_4, layers_5, layers_6], 2)
    outputs = tf.keras.layers.Flatten()(layers)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.trainable = False
    return model

def InpactorFNN_Net(input_dim):
    tf.keras.backend.clear_session()

    inputs = tf.keras.Input(shape=(input_dim,), name="input_1")
    
    layers = tf.keras.layers.Dense(400, activation="relu")(inputs)
    layers = tf.keras.layers.Dropout(0.2)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    
    layers = tf.keras.layers.Dense(400, activation="relu")(layers)
    layers = tf.keras.layers.Dropout(0.2)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)

    layers = tf.keras.layers.Dense(400, activation="relu")(layers)
    layers = tf.keras.layers.Dropout(0.2)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)
    
    layers = tf.keras.layers.Dense(400, activation="relu")(layers)
    layers = tf.keras.layers.Dropout(0.2)(layers)
    layers = tf.keras.layers.BatchNormalization()(layers)

    # Output con 3 classes
    predictions = tf.keras.layers.Dense(3, activation="softmax", name="output_1")(layers)
    
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    
    model.compile(loss=loss_fn, optimizer=opt, metrics=[f1_m, 'accuracy'])
    return model

class InpactorDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, sequences, labels=None, batch_size=32, kmer_model=None, scaler=None, pca=None, shuffle=True, num_classes=3):
        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.kmer_model = kmer_model
        self.scaler = scaler
        self.pca = pca
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.indexes = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.sequences) / self.batch_size)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.sequences))
        batch_indexes = self.indexes[start_index:end_index]

        batch_sequences = [self.sequences[i] for i in batch_indexes]
        
        # k-mers
        kmer_counts = []
        for seq in batch_sequences:
            one_hot = fasta2one_hot(str(seq.seq), len(seq))
            model_input = np.expand_dims(one_hot, axis=-1).astype(np.float32)
            # usar cpu para evitar problemas de memoria
            with tf.device('/CPU:0'):
                counts = self.kmer_model(model_input, training=False)
            kmer_counts.append(counts[0].numpy())
            
        kmer_counts = np.array(kmer_counts)
        
        scaled_features = self.scaler.transform(kmer_counts)
        
        # PCA
        pca_features = self.pca.transform(scaled_features)
        
        X = pca_features

        if self.labels is not None:
            y = self.labels[batch_indexes]
            return X, to_categorical(y, num_classes=self.num_classes)
        else:
            return X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

class InpactorModel:
    def __init__(self, model_path=None):
        self.installation_path = os.path.dirname(os.path.realpath(__file__))
        self.scaler_path = os.path.join(self.installation_path, 'Models', 'std_scaler_filter.bin')
        self.pca_path = os.path.join(self.installation_path, 'Models', 'std_pca_filter.bin')
        
        print("Cargando Scaler y PCA...")
        self.scaler = load(self.scaler_path)
        self.pca = load(self.pca_path)
        
        self.kmer_model = kmer_extractor_model((5, None, 1))
        
        self.input_dim = self.pca.n_components_
        
        if model_path and os.path.exists(model_path):
            print(f"Cargando modelo desde {model_path}...")
            self.model = tf.keras.models.load_model(model_path, custom_objects={'f1_m': f1_m})
        else:
            print("Creando nuevo modelo FNN")
            self.model = InpactorFNN_Net(self.input_dim)
    
    def prepare_data(self, fasta_file):
        print(f"Leyendo secuencias desde {fasta_file}...")
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        if not sequences:
            print(f"No secuencias encontradas en {fasta_file}.")
            return None
        return sequences

    def prepare_training_data(self, fasta_file):
        print(f"Leyendo secuencias desde {fasta_file}...")
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        if not sequences:
            print(f"No sequences found in {fasta_file}.")
            return None, None
        
        y = []
        for seq in sequences:
            seq_id = seq.id
            if not seq_id.startswith("Caso"):
                y.append(1) # Kept
            else:
                if seq_id.startswith("Caso4") or seq_id.startswith("Caso5"):
                    y.append(2) # No_TE
                elif seq_id.startswith("Caso1") or seq_id.startswith("Caso2") or seq_id.startswith("Caso3"):
                    y.append(0) # Removed
                else:
                    print(f"Warning: Caso desconocido para {seq_id}, default 0")
                    y.append(0)
        y = np.array(y)

        return sequences, y

    def train(self, fasta_file, output_dir, epochs=50, batch_size=32):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sequences, y = self.prepare_training_data(fasta_file)
        if sequences is None: return
        
        # 70% train, 15% test, 15% val
        seq_temp, seq_test, y_temp, y_test = train_test_split(sequences, y, test_size=0.15, random_state=42, stratify=y)
        
        seq_train, seq_val, y_train, y_val = train_test_split(seq_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
        
        print(f"Training set: {len(seq_train)} sequences")
        print(f"Validation set: {len(seq_val)} sequences")
        print(f"Test set: {len(seq_test)} sequences")
        
        # Generators
        train_gen = InpactorDataGenerator(seq_train, y_train, batch_size=batch_size, 
                                          kmer_model=self.kmer_model, scaler=self.scaler, pca=self.pca, shuffle=True)
        val_gen = InpactorDataGenerator(seq_val, y_val, batch_size=batch_size, 
                                        kmer_model=self.kmer_model, scaler=self.scaler, pca=self.pca, shuffle=False)
        test_gen = InpactorDataGenerator(seq_test, y_test, batch_size=batch_size, 
                                         kmer_model=self.kmer_model, scaler=self.scaler, pca=self.pca, shuffle=False)
        
        # Callbacks
        output_model_path = os.path.join(output_dir, "trained_model.h5")
        log_dir = os.path.join(output_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
            tf.keras.callbacks.ModelCheckpoint(output_model_path, monitor='val_f1_m', save_best_only=True, mode='max', verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        ]
        
        print("\TRAIN ...")
        self.model.fit(train_gen, 
                       epochs=epochs, 
                       validation_data=val_gen,
                       callbacks=callbacks,
                       verbose=1)
        
        self.model.save(output_model_path)
        print(f"Modelo guardado en {output_model_path}")
        
        print("EVALUATION CON TEST")
        best_model = tf.keras.models.load_model(output_model_path, custom_objects={'f1_m': f1_m})
        
        y_pred_probs = best_model.predict(test_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        target_names = ['Removed (0)', 'Kept (1)', 'No_TE (2)']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Test Set)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_test.png'))
        print(f"Confusion matrix guardada en {os.path.join(output_dir, 'confusion_matrix_test.png')}")

    def predict(self, fasta_file, output_file=None):
        sequences = self.prepare_data(fasta_file)
        if sequences is None: return
        
        pred_gen = InpactorDataGenerator(sequences, labels=None, batch_size=32, 
                                         kmer_model=self.kmer_model, scaler=self.scaler, pca=self.pca, shuffle=False)
        
        print("PREDICCIÓN...")
        predictions = self.model.predict(pred_gen)
        predicted_classes = np.argmax(predictions, axis=1)
        
        results = []
        for i, seq in enumerate(sequences):
            pred_class = predicted_classes[i]
            confidence = predictions[i][pred_class]
            
            status = "Unknown"
            if pred_class == 0:
                status = "Removed (Needs curation)"
            elif pred_class == 1:
                status = "Kept (No curation needed)"
            elif pred_class == 2:
                status = "No TE"
                
            results.append((seq.id, pred_class, confidence, status))
            
        if output_file:
            with open(output_file, 'w') as f:
                f.write("ID\tClass\tConfidence\tStatus\n")
                for res in results:
                    f.write(f"{res[0]}\t{res[1]}\t{res[2]:.4f}\t{res[3]}\n")
            print(f"Resultados guardados en {output_file}")
        else:
            for res in results:
                print(f"{res[0]}\t{res[1]}\t{res[2]:.4f}\t{res[3]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpactor2 Model for TE Curation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict")
    predict_parser.add_argument("-f", "--fasta", required=True, help="fichero FASTA")
    predict_parser.add_argument("-m", "--model", help="fichero del modelo")
    predict_parser.add_argument("-o", "--output", help="fichero de salida")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train")
    train_parser.add_argument("-i", "--input", required=True, help="fichero FASTA")
    train_parser.add_argument("-o", "--output", required=True, help="directorio de salida")
    train_parser.add_argument("--epochs", type=int, default=50, help="Número de epochs")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Tamaño del batch")
    
    args = parser.parse_args()
    
    if args.command == "predict":
        if not os.path.exists(args.fasta):
            print(f"Error:fichero {args.fasta} no encontrado.")
            sys.exit(1)
        inpactor = InpactorModel(args.model)
        inpactor.predict(args.fasta, args.output)
        
    elif args.command == "train":
        if not os.path.exists(args.input):
            print(f"Error: fichero {args.input} no encontrado.")
            sys.exit(1)
                
        inpactor = InpactorModel()
        inpactor.train(args.input, args.output, args.epochs, args.batch_size)
    
    else:
        parser.print_help()
