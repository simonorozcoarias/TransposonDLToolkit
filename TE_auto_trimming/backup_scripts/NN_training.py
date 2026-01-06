# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "default"

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

from pickle import dump, load
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
import numpy as np
import re
#from pdf2image import convert_from_path
#from pdf2image.exceptions import PDFPageCountError
import cv2
from focal_loss import BinaryFocalLoss
import sys
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import r2_score
import fitz

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# para crear mapa de flujo pydot y graphviz
try:
    import pydot
except ImportError:
    print("pydot no esta instalado. Instalando...")
    os.system('pip install pydot')
    import pydot

try:
    import graphviz
except ImportError:
    print("graphviz no esta instalado. Instalando...")
    os.system('pip install graphviz')
    import graphviz


class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
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
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X

    def save_model(self, model_name):
        dump(self._scaler, open(model_name + '.bin', 'wb'))

    def load_model(self, model_path, X):
        self._scaler = load(open(model_path, 'rb'))
        self._orig_shape = X.shape[1:]

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        tensor_shape2 = images.get_shape()
        print(tensor_shape2)
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        print(patches.shape)
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Implementacion tipica de un Vision Transformer (ViT). 
# Sirve para codificar los patches extraidos de una imagen, antes de enviarlos a capas tipo Transformer.
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def recall_m(y_true, y_pred):
    # K es un alias comunmente usado para keras.backend
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def r2_score(y_true, y_pred):
    # Suma de los cuadrados de los residuos
    sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred))

    # Media de los valores reales
    mean_y_true = tf.reduce_mean(y_true)

    # Suma de los cuadrados totales
    sum_squares = tf.reduce_sum(tf.square(y_true - mean_y_true))

    # Evitar division por cero ESTO LO HE PUESTO YO PGS
    epsilon = tf.keras.backend.epsilon()
    # sum_squares = tf.maximum(sum_squares, epsilon)

    # Coeficiente de determinacion R2
    R2 = 1 - sum_squares_residuals / sum_squares
    return R2


def metrics(Y_validation, predictions, name):
    classes = len(np.unique(Y_validation))
    print('Accuracy:', accuracy_score(Y_validation, predictions))
    print('F1 score:', f1_score(Y_validation, predictions, average='weighted'))
    print('Recall:', recall_score(Y_validation, predictions, average='weighted'))
    print('Precision:', precision_score(Y_validation, predictions, average='weighted'))
    print('Specificity:', recall_score(Y_validation, predictions, pos_label=0, average='weighted'))
    print('\n clasification report:\n', classification_report(Y_validation, predictions))
    print('\n confusion matrix:\n', confusion_matrix(Y_validation, predictions))
    # Creamos la matriz de confusion
    snn_cm = confusion_matrix(Y_validation, predictions)

    # Visualizamos la matriz de confusion
    snn_df_cm = pd.DataFrame(snn_cm, range(classes), range(classes))
    plt.figure(figsize=(20, 14))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(snn_df_cm, annot=True, annot_kws={"size": 25}, fmt='', cmap="YlOrRd")  # font size
    plt.savefig('confusion_matrix_' + name, bbox_inches='tight', dpi=500)
    # plt.show()


def write_sequences_file(sequences, filename):
    try:
        SeqIO.write(sequences, filename, "fasta")
    except FileNotFoundError:
        print("FATAL ERROR: I couldn't find the file, please check: '" + filename + "'. Path not found")
        sys.exit(0)
    except PermissionError:
        print("FATAL ERROR: I couldn't access the files, please check: '" + filename + "'. I don't have permissions.")
        sys.exit(0)
    except Exception as exp:
        print("FATAL ERROR: There is a unknown problem writing sequences in : '" + filename + "'.")
        print(exp)
        sys.exit(0)

# Crea un dataset en matrices numpy a partir de la lista de TEs y de sus plots
# Tambien genera una matriz con las etiquetas con la informaci�n de posiciones de inicio y final
# Esto se hace para poder predecir estas posiciones a partir de la informacion de los plots

def count_good_pdfs(MCH_out):
    pdf_dir = os.path.join(MCH_out, "te_aid")
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    count_good = 0
    count_bad = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        try:
            # Verifica tama�o > 4 KB
            if os.path.getsize(pdf_path) <= 4 * 1024:
                count_bad += 1
                continue

            # Intenta abrir el PDF
            doc = fitz.open(pdf_path)
            if len(doc) > 0:
                count_good += 1
            else:
                count_bad += 1

        except Exception:
            count_bad += 1

    return count_good, count_bad

def create_dataset(library, MCH_out, outputDir):
    # Lee el archivo fasta library y devuelve una lista de objetos SeqRecord con atributos como TE.id y TE.seq
    TEs = [TE for TE in SeqIO.parse(library, "fasta")]
    
    good_pdfs, bad_pdfs = count_good_pdfs(MCH_out)
    
    # Para cada secuencia TE, crea una matriz
    feature_data1 = np.zeros((good_pdfs, 256, 256, 4), dtype=np.uint8)
    # Crea una matriz para almacenar las categorias como etiquetas
    labels = np.zeros((good_pdfs, 2), dtype=np.float32)
    case_names = []
    species_names = []
    
    print(good_pdfs, bad_pdfs)

    n = 0
    for TE in TEs:

        print("Doing TE.id: " + TE.id)
        TE_name = TE.id.split("#")[0]
        
        print("Doing TE_name: " + TE.id)
        case_id = TE.id.split("_")[0]
        
        print("Doing case_id: " + TE.id)
        species_match = re.search(r'([A-Z][a-z]+_[a-z]+)$', TE.id)
        species_name = species_match.group(0) if species_match else None
            
        print(f"Doing TE_name: {TE_name}")
        try:

            pdf_path = MCH_out + '/te_aid/' + TE_name + '.pdf'
            
            if not os.path.exists(pdf_path):
                print(f"PDF no encontrado para {TE_name}, se omite.")
                continue
    
            if os.path.getsize(pdf_path) <= 4 * 1024:
                continue
                
            doc = fitz.open(pdf_path)

            # Saving pages in jpeg format
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(200/72,200/72))
                pix.save(MCH_out + '/te_aid/' + TE_name + '.fa.c2g.jpeg', 'JPEG')

            # Carga la imagen desde el archivo
            te_aid_image = cv2.imread(MCH_out + '/te_aid/' + TE_name + '.fa.c2g.jpeg')
            te_aid_image = cv2.cvtColor(te_aid_image, cv2.COLOR_BGR2GRAY)
            
            if te_aid_image is None:
                print("ERROR: No se pudo abrir la imagen de", TE_name)
                continue  # pasa al siguiente TE
                
            h, w = te_aid_image.shape
            print(f"{TE_name}: h={h}, w={w}")

            # Divide la imagen en distintos plots y les hace resize
            divergence_plot = cv2.resize(te_aid_image[150:1030, 150:1130], (256, 256))
            coverage_plot = cv2.resize(te_aid_image[150:1030, 1340:2320], (256, 256))
            selfdot_plot = cv2.resize(te_aid_image[1340:2220, 150:1130], (256, 256))
            structure_plot = cv2.resize(te_aid_image[1340:2220, 1340:2320], (256, 256))
            
            """plt.xticks([]), plt.yticks([])
            plt.imshow(structure_plot, cmap='gray', interpolation='bicubic')
            plt.show()"""

            # Guarda la informacion de los plots en la matriz feature_data1
            feature_data1[n, :, :, 0] = divergence_plot
            feature_data1[n, :, :, 1] = coverage_plot
            feature_data1[n, :, :, 2] = selfdot_plot
            feature_data1[n, :, :, 3] = structure_plot

            # Guarda el penultimo valor del TE.id (posici�n de inicio)
            start_pos = int(TE.id.split("_")[-4])
            print(f"start_pos: {start_pos}")

            # Guarda el ultimo valor del TE.id (longitud del TE)
            TE_len = int(TE.id.split("_")[-3])
            
            print(f"TE_len: {TE_len}")

            # Guarda en las etiquetas la posicion de inicio y la de final, relativa a la longitud total
            labels[n, 0] = start_pos / 15000
            
            end_pos = start_pos + TE_len
            
            if end_pos > 15000:
                labels[n, 1] = 1
                
            else:
                labels[n, 1] = end_pos / 15000
            
            case_names.append(TE.id)
            species_names.append(species_name)
            
            print(f"n: {n}")
            n += 1
        except Exception as ex:
            print("something wrong with " + TE_name)
            print(ex)

    np.save(outputDir + "/features_data.npy", feature_data1)
    np.save(outputDir + "/labels_data.npy", labels)
    np.save(outputDir + "/case_labels.npy", np.array(case_names))
    np.save(outputDir + "/species_labels.npy", np.array(species_names))

# Creamos un modelo por cada rama para procesar tipos distintos de features (como divergence, coverage, etc.)
# Despues se generara un modelo completo que agrupe a todas las branches CNN
def cnn_branch(dim1, dim2, dim3, i):
    tf.keras.backend.clear_session()

    # Inputs
    inputs = tf.keras.Input(shape=(dim1, dim2, dim3), name="input_" + str(i))

    # layer 1
    layers = tf.keras.layers.Conv2D(16, (5, 5), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
                                    bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
                                    padding="same", name="conv_" + str(i) + "_1")(inputs)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_2d_" + str(i) + "_1")(layers)
    layers = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, scale=False,
                                                name="BN_" + str(i) + "_1")(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, name="max_pool_" + str(i) + "_1")(layers)

    # layer 2
    layers = tf.keras.layers.Conv2D(32, (5, 5), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
                                    bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
                                    padding="same", name="conv_" + str(i) + "_2")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_2d_" + str(i) + "_2")(layers)
    layers = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.4, epsilon=0.001, scale=False,
                                                name="BN_" + str(i) + "_2")(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, name="max_pool_" + str(i) + "_2")(layers)

    # layer 3
    layers = tf.keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
                                    bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
                                    padding="same", name="conv_" + str(i) + "_3")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_2d_" + str(i) + "_3")(layers)
    layers = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.2, epsilon=0.001, scale=False,
                                                name="BN_" + str(i) + "_3")(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, name="max_pool_" + str(i) + "_3")(layers)

    # layer 4
    layers = tf.keras.layers.Conv2D(128, (5, 5), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                    kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
                                    bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
                                    padding="same", name="conv_" + str(i) + "_4")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_2d_" + str(i) + "_4")(layers)
    layers = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.2, epsilon=0.001, scale=False,
                                                name="BN_" + str(i) + "_4")(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=None, name="max_pool_" + str(i) + "_4")(layers)

    # layer 5
    layers = tf.keras.layers.Flatten(name="flatten_" + str(i) + "_1")(layers)

    # layer end
    predictions = tf.keras.layers.Dense(128, activation="sigmoid", name="outputF_" + str(i))(layers)
    # model generation
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    return model

# Crea el modelo a partir de las distintas branches / features creadas con cnn_branch 
def auto_trimming(cnn_div, cnn_cov, cnn_dot, cnn_str):
    combinedInput = tf.keras.layers.concatenate([cnn_div.output, cnn_cov.output, cnn_dot.output, cnn_str.output])

    # layer 1
    layers = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001),
                                   name="dense_5_1")(combinedInput)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_1")(layers)
    #layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True,
    #                                            fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4,
    #                                            adjustment=None)(layers)
    
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True)(layers)
    
    # layer 2
    layers = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.01),
                                   name="dense_5_2")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_2")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True)(layers)
    # layer 3
    layers = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.01),
                                   name="dense_5_3")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_3")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True)(layers)
    # layer 4
    layers = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001),
                                   name="dense_5_4")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_4")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True)(layers)
    # layer 5
    layers = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001),
                                   name="dense_5_5")(layers)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_5")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True)(layers)

    # layer 6
    predictions = tf.keras.layers.Dense(2, activation="sigmoid", name="output_5")(layers)

    model = tf.keras.Model(inputs=[cnn_div.input, cnn_cov.input, cnn_dot.input, cnn_str.input], outputs=predictions)
    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # Compile model
    model.compile(loss="mse", optimizer=opt, metrics=[r2_score])
    return model


def testing_models2(model_path, scalerx_path, dataX_path, dataY_path):
    X_test = np.load(dataX_path)
    Y_test = np.load(dataY_path).astype(np.float32)

    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    print("NaNs in X_test:", np.isnan(X_test).sum())
    print("NaNs in Y_test:", np.isnan(Y_test).sum())

    """scalerX = NDStandardScaler()
                scalerX.load_model(scalerx_path, X_test)
            
                X1_test_scl = scalerX.transform(X_test)"""

    X1_test_scl = X_test
    print("NaNs in X1_test_scl after scaling:", np.isnan(X1_test_scl).sum())

    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU(0.1),
                                                       'r2_score': r2_score})

    predictions = model.predict(
        [X1_test_scl[:, :, :, 0], X1_test_scl[:, :, :, 1], X1_test_scl[:, :, :, 2],
         X1_test_scl[:, :, :, 3]], verbose=0)

    predictions = np.nan_to_num(predictions, nan=0)

    print("NaNs in predictions:", np.isnan(predictions).sum())
    print("predictions shape:", predictions.shape)
    print(Y_test[:, 0])
    print(predictions[:, 0])

    r2_initial = calcular_y_graficar_r23(Y_test[:, 0], predictions[:, 0], "StartingPos")
    r2_final = calcular_y_graficar_r23(Y_test[:, 1], predictions[:, 1], "EndingPos")
    print("R2 starting position:" + str(r2_initial))
    print("R2 ending position:" + str(r2_final))


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


def run_experiment(model, X_train, Y_train, X_dev, Y_dev, batch_size, num_epochs):
    # Verifica que no haya valores NaN en los datasets
    if np.isnan(X_train).any() or np.isnan(Y_train).any() or np.isnan(X_dev).any() or np.isnan(Y_dev).any():
        print("hay valores nan")
        return None, None
        
    # Crear carpeta "checkpoint" si no existe
    checkpoint_dir = "./checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_filepath = os.path.join(checkpoint_dir, "model.weights.h5")

    # Define el callback ReduceLROnPlateau
    # Reduce la tasa de aprendizaje si el val_loss no mejora durante 10 epocas
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=10, verbose=1)

    # Guarda los pesos del modelo con mejor val_r2_score
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_r2_score",
        save_best_only=True,
        save_weights_only=True,
    )

    # Detiene el entrenamiento si val_loss no mejora durante 50 epocas
    # Restaura los pesos del modelo que obtuvo el mejor val_loss
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )

    # Entrena el modelo con los datos de entrenamiento
    history = model.fit(
        x=X_train,
        y=Y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[checkpoint_callback, lr_scheduler, early_stopping],
        validation_data=(X_dev, Y_dev), verbose=1)

    return history, checkpoint_callback


if __name__ == '__main__':
    option = sys.argv[1]

    if option == "dataset":
        if len(sys.argv) == 5:
            library = sys.argv[2]
            MCH_out = sys.argv[3]
            outputDir = sys.argv[4]
            create_dataset(library, MCH_out, outputDir)
        else:
            print(
                "ERROR in parameters, usage: python3 SmartInspection.py dataset TE_library.fasta path/to/MCHelper/output path/to/output")
            sys.exit(0)

    elif option == "train":
        if len(sys.argv) == 4:

            learning_rate = 0.0005
            weight_decay = 0.0001
            batch_size = 16
            num_epochs = 200

            x_str = sys.argv[2]
            Y_str = sys.argv[3]

            # Load data
            x = np.load(x_str) / 255.0
            x = x.astype(np.float16)
            print(f"Loaded X: {x.shape}")
            y = np.load(Y_str)
            print(f"Loaded Y: {y.shape}")

            # Split en train/dev/test
            validation_size = 0.2
            seed = 7
            X_train, X_test_dev, Y_train, Y_test_dev = train_test_split(x, y, test_size=validation_size,
                                                                        random_state=seed)
            X_dev, X_test, Y_dev, Y_test = train_test_split(X_test_dev, Y_test_dev, test_size=0.5, random_state=seed)
            
            # Scaler normalization
            scalerX = NDStandardScaler().fit(X_train)
            X_train = scalerX.transform(X_train)
            X_dev = scalerX.transform(X_dev)
            X_test = scalerX.transform(X_test)

            scalerX.save_model("scalerX")
            
            # Información de los datasets
            print("Information about the data")
            print(f"Training shape: X={X_train.shape}, Y={Y_train.shape}")
            print(f"Dev shape: X={X_dev.shape}, Y={Y_dev.shape}")
            print(f"Test shape: X={X_test.shape}, Y={Y_test.shape}")

            # Generate the NNs
            cnn_div = cnn_branch(X_train.shape[1], X_train.shape[2], 1, 1)
            cnn_cov = cnn_branch(X_train.shape[1], X_train.shape[2], 1, 2)
            cnn_dot = cnn_branch(X_train.shape[1], X_train.shape[2], 1, 3)
            cnn_str = cnn_branch(X_train.shape[1], X_train.shape[2], 1, 4)
            
            # Final model
            model = auto_trimming(cnn_div, cnn_cov, cnn_dot, cnn_str)
            print(model.summary())
            #tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

            # Separate channels
            path_log_base = os.path.dirname(os.path.realpath(__file__))
            
            training_set = [
                X_train[:, :, :, 0:1], 
                X_train[:, :, :, 1:2], 
                X_train[:, :, :, 2:3],
                X_train[:, :, :, 3:4]
            ]
            
            dev_set = [
                X_dev[:, :, :, 0:1], 
                X_dev[:, :, :, 1:2], 
                X_dev[:, :, :, 2:3], 
                X_dev[:, :, :, 3:4]
            ]
            
            test_set = [
                X_test[:, :, :, 0:1], 
                X_test[:, :, :, 1:2], 
                X_test[:, :, :, 2:3], 
                X_test[:, :, :, 3:4]
            ]

            # Run training
            history, checkpoints = run_experiment(
                model, training_set, Y_train, dev_set, Y_dev, batch_size, num_epochs)

            acc = history.history['r2_score']
            np.save('accuracy_SENMAP_completeDB.npy', acc)
            val_acc = history.history['val_r2_score']
            np.save('val_accuracy_SENMAP_completeDB.npy', val_acc)
            loss = history.history['loss']
            np.save('loss_SENMAP_completeDB.npy', loss)
            val_loss = history.history['val_loss']
            np.save('val_los_SENMAP_completeDBnpy', val_loss)

            # === Plot R2 ===
            plt.figure()
            plt.plot(history.history['r2_score'])
            plt.plot(history.history['val_r2_score'])
            plt.legend(['train_r2_score', 'val_r2_score'], loc='lower right')
            plt.xlabel('Epoch')
            plt.ylabel('R2 Score')
            plt.title('Epoch vs R2 Score')
            plt.savefig('Train_Curve_R2.png', bbox_inches='tight', dpi=500)
                       
            # === Plot Loss (MSE) ===
            plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.legend(['train_loss', 'val_loss'], loc='upper right')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.title('Epoch vs Loss')
            plt.savefig('Train_Curve_Loss_bottom.png', bbox_inches='tight', dpi=500)
            
            model.save('trained_model.h5')
            np.save("X_test.npy", X_test)
            np.save("Y_test.npy", Y_test)
        else:
            print("ERROR in parameters, usage: python3 NN_trainingV2.py train X Y")
            sys.exit(0)

    elif option == "train_batches":
        if len(sys.argv) == 11:
            X1_train_str = sys.argv[2]
            X2_train_str = sys.argv[3]
            Y_train_str = sys.argv[4]
            X1_dev_str = sys.argv[5]
            X2_dev_str = sys.argv[6]
            Y_dev_str = sys.argv[7]
            X1_test_str = sys.argv[8]
            X2_test_str = sys.argv[9]
            Y_test_str = sys.argv[10]
            num_batches = 3

            X1_train = np.load(X1_train_str + "_batch_0.npy")
            # X1_train = X1_train / 255.0
            # X1_train = X1_train.astype(np.float16)
            X2_train = np.load(X2_train_str + "_batch_0.npy")
            Y_train = np.load(Y_train_str + "_batch_0.npy")
            print("Train dataset loaded ...")
            X1_dev = np.load(X1_dev_str + "_batch_0.npy")
            # X1_dev = X1_dev / 255.0
            # X1_dev = X1_dev.astype(np.float16)
            X2_dev = np.load(X2_dev_str + "_batch_0.npy")
            Y_dev = np.load(Y_dev_str + "_batch_0.npy")
            print("Dev dataset loaded ...")
            X1_test = np.load(X1_test_str + "_batch_0.npy")
            # X1_test = X1_test / 255.0
            # X1_test = X1_test.astype(np.float16)
            X2_test = np.load(X2_test_str + "_batch_0.npy")
            Y_test = np.load(Y_test_str + "_batch_0.npy")
            print("Test dataset loaded ...")

            """scalerX1 = NDStandardScaler().fit(X1_train)
            X1_train = scalerX1.transform(X1_train)
            scalerX2 = NDStandardScaler().fit(X2_train)
            X2_train = scalerX2.transform(X2_train)
            X1_dev = scalerX1.transform(X1_dev)
            X2_dev = scalerX2.transform(X2_dev)
            X1_test = scalerX1.transform(X1_test)
            X2_test = scalerX2.transform(X2_test)
            print("scaling done ...")

            scalerX1.save_model("scalerX1")
            scalerX2.save_model("scalerX2")"""

            print("Information about the data")
            print("Training shape: X1=" + str(X1_train.shape) + ", X2=" + str(X2_train.shape) + ", Y=" + str(
                Y_train.shape) + ", Positive=" + str(Y_train[Y_train == 0].shape[0]) + " (" + str(
                Y_train[Y_train == 0].shape[0] / Y_train.shape[0]) + ")")
            print("Dev shape: X1=" + str(X1_dev.shape) + ", X2=" + str(X2_dev.shape) + ", Y=" + str(
                Y_dev.shape) + ", Positive=" + str(Y_dev[Y_dev == 0].shape[0]) + " (" + str(
                Y_dev[Y_dev == 0].shape[0] / Y_dev.shape[0]) + ")")
            print("Test shape: X1=" + str(X1_test.shape) + ", X2=" + str(X2_test.shape) + ", Y=" + str(
                Y_test.shape) + ", Positive=" + str(Y_test[Y_test == 0].shape[0]) + " (" + str(
                Y_test[Y_test == 0].shape[0] / Y_test.shape[0]) + ")")

            # generate the NNs
            cnn_div = cnn_branch(X1_train.shape[1], X1_train.shape[2], X1_train.shape[3], 1)
            cnn_cov = cnn_branch(X1_train.shape[1], X1_train.shape[2], X1_train.shape[3], 2)
            cnn_str = cnn_branch(X1_train.shape[1], X1_train.shape[2], X1_train.shape[3], 3)
            fnn = cnn_branch(X2_train.shape[1]) # error corregido fnn_branch -> cnn_branch
            model = SmartInspection(cnn_div, cnn_cov, cnn_str, fnn) # SmartInspection es un tipo de clase
            # summarize layers
            print(model.summary())
            #tf.keras.utils.plot_model(model, show_shapes=True)

            path_log_base = os.path.dirname(os.path.realpath(__file__))
            epochs = 150
            batch_size = 128
            """training_set = [X1_train[:, :, :, 0], X1_train[:, :, :, 1], X1_train[:, :, :, 2], X2_train]
            dev_set = [X1_dev[:, :, :, 0], X1_dev[:, :, :, 1], X1_dev[:, :, :, 2], X2_dev]
            test_set = [X1_test[:, :, :, 0], X1_test[:, :, :, 1], X1_test[:, :, :, 2], X2_test]"""
            print("Model loaded ...")
            results = train(path_log_base, model,
                            [X1_train[:, :, :, 0], X1_train[:, :, :, 1], X1_train[:, :, :, 2], X2_train], Y_train,
                            [X1_dev[:, :, :, 0], X1_dev[:, :, :, 1], X1_dev[:, :, :, 2], X2_dev], Y_dev,
                            [X1_test[:, :, :, 0], X1_test[:, :, :, 1], X1_test[:, :, :, 2], X2_test], Y_test,
                            batch_size, epochs, "SmartInspection")

            # Final_Results_Test(log_Dir, test_set, Y_test)
            # print("results: "+results)
            for i in range(1, num_batches):
                X1_train = np.load(X1_train_str + "_batch_" + str(i) + ".npy")
                # X1_train = X1_train / 255.0
                # X1_train = X1_train.astype(np.float16)
                X2_train = np.load(X2_train_str + "_batch_" + str(i) + ".npy")
                Y_train = np.load(Y_train_str + "_batch_" + str(i) + ".npy")
                print("Train dataset loaded ...")
                X1_dev = np.load(X1_dev_str + "_batch_" + str(i) + ".npy")
                # X1_dev = X1_dev / 255.0
                # X1_dev = X1_dev.astype(np.float16)
                X2_dev = np.load(X2_dev_str + "_batch_" + str(i) + ".npy")
                Y_dev = np.load(Y_dev_str + "_batch_" + str(i) + ".npy")
                print("Dev dataset loaded ...")
                X1_test = np.load(X1_test_str + "_batch_" + str(i) + ".npy")
                # X1_test = X1_test / 255.0
                # X1_test = X1_test.astype(np.float16)
                X2_test = np.load(X2_test_str + "_batch_" + str(i) + ".npy")
                Y_test = np.load(Y_test_str + "_batch_" + str(i) + ".npy")
                print("Test dataset loaded ...")
                results = train(path_log_base, model,
                                [X1_train[:, :, :, 0], X1_train[:, :, :, 1], X1_train[:, :, :, 2], X2_train], Y_train,
                                [X1_dev[:, :, :, 0], X1_dev[:, :, :, 1], X1_dev[:, :, :, 2], X2_dev], Y_dev,
                                [X1_test[:, :, :, 0], X1_test[:, :, :, 1], X1_test[:, :, :, 2], X2_test], Y_test,
                                batch_size, epochs, "SmartInspection")

            print("Metrics training")
            predictions = model.predict([X1_train[:, :, :, 0], X1_train[:, :, :, 1], X1_train[:, :, :, 2], X2_train])
            metrics(Y_train, predictions > 0.6, "train")
            scores = model.evaluate([X1_train[:, :, :, 0], X1_train[:, :, :, 1], X1_train[:, :, :, 2], X2_train],
                                    Y_train, verbose=0)
            print("Baseline Error train: %.2f%%" % (100 - scores[1] * 100))

            print("Metrics dev")
            predictions = model.predict([X1_dev[:, :, :, 0], X1_dev[:, :, :, 1], X1_dev[:, :, :, 2], X2_dev])
            metrics(Y_dev, predictions > 0.6, "dev")
            scores = model.evaluate([X1_dev[:, :, :, 0], X1_dev[:, :, :, 1], X1_dev[:, :, :, 2], X2_dev], Y_dev,
                                    verbose=0)
            print("Baseline Error dev: %.2f%%" % (100 - scores[1] * 100))

            print("Metrics test")
            predictions = model.predict([X1_test[:, :, :, 0], X1_test[:, :, :, 1], X1_test[:, :, :, 2], X2_test])
            metrics(Y_test, predictions > 0.6, "test")
            scores = model.evaluate([X1_test[:, :, :, 0], X1_test[:, :, :, 1], X1_test[:, :, :, 2], X2_test], Y_test,
                                    verbose=0)
            print("Baseline Error test: %.2f%%" % (100 - scores[1] * 100))
        else:
            print(
                "ERROR in parameters, usage: python3 SmartInspection.py train X1_train.npy X2_train.npy Y_train.npy X1_dev.npy X2_dev.npy Y_dev.npy X1_test.npy X2_test.npy Y_test.npy")
            sys.exit(0)
    elif option == "scaling":
        if len(sys.argv) == 11:
            X1_train_str = sys.argv[2]
            X2_train_str = sys.argv[3]
            Y_train_str = sys.argv[4]
            X1_dev_str = sys.argv[5]
            X2_dev_str = sys.argv[6]
            Y_dev_str = sys.argv[7]
            X1_test_str = sys.argv[8]
            X2_test_str = sys.argv[9]
            Y_test_str = sys.argv[10]

            X1_train = np.load(X1_train_str)
            X2_train = np.load(X2_train_str)
            Y_train = np.load(Y_train_str)
            X1_dev = np.load(X1_dev_str)
            X2_dev = np.load(X2_dev_str)
            Y_dev = np.load(Y_dev_str)
            X1_test = np.load(X1_test_str)
            X2_test = np.load(X2_test_str)
            Y_test = np.load(Y_test_str)

            scalerX1 = NDStandardScaler().fit(X1_train)
            X1_train = scalerX1.transform(X1_train)
            scalerX2 = NDStandardScaler().fit(X2_train)
            X2_train = scalerX2.transform(X2_train)
            X1_dev = scalerX1.transform(X1_dev)
            X2_dev = scalerX2.transform(X2_dev)
            X1_test = scalerX1.transform(X1_test)
            X2_test = scalerX2.transform(X2_test)

            scalerX1.save_model("scalerX1")
            scalerX2.save_model("scalerX2")

            np.save(X1_train_str + "_scaled.npy", X1_train.astype(np.float16))
            np.save(X2_train_str + "_scaled.npy", X2_train.astype(np.float16))

            np.save(X1_dev_str + "_scaled.npy", X1_dev.astype(np.float16))
            np.save(X2_dev_str + "_scaled.npy", X2_dev.astype(np.float16))

            np.save(X1_test_str + "_scaled.npy", X1_test.astype(np.float16))
            np.save(X2_test_str + "_scaled.npy", X2_test.astype(np.float16))
    elif option == "batch_creation":
        X1_train_str = sys.argv[2]
        X2_train_str = sys.argv[3]
        Y_train_str = sys.argv[4]

        X1_train = np.load(X1_train_str)
        X2_train = np.load(X2_train_str)
        Y_train = np.load(Y_train_str)

        # Definir el numero de batches
        num_bat = 3
        row_per_batc = int(X1_train.shape[0] / num_bat)

        # Aleatorizar las muestras
        indices = np.arange(X1_train.shape[0])
        np.random.shuffle(indices)

        # Dividir los datos en batches aleatorios y guardarlos
        for b in range(num_bat):
            start = row_per_batc * b
            end = row_per_batc * (b + 1)
            if end >= X1_train.shape[0]:
                end = X1_train.shape[0]

            batch_indices = indices[start:end]
            np.save(X1_train_str + "_batch_" + str(b) + ".npy", X1_train[batch_indices].astype(np.float16))
            np.save(X2_train_str + "_batch_" + str(b) + ".npy", X2_train[batch_indices].astype(np.float16))
            np.save(Y_train_str + "_batch_" + str(b) + ".npy", Y_train[batch_indices].astype(np.float16))

    elif option == "test":
        if len(sys.argv) == 6:
            model_path = sys.argv[2]
            scalerx1_path = sys.argv[3]
            dataX1_path = sys.argv[4]
            dataY_path = sys.argv[5]
            testing_models2(model_path, scalerx1_path, dataX1_path, dataY_path)
        else:
            print(
                "ERROR in parameters, usage: python3 SmartInspection.py test model_path scalerX1_path dataX1_path dataY_path data_TE_names_path")
            sys.exit(0)
    else:
        print("Option not found: " + option)
