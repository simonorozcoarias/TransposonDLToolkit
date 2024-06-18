from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, \
    classification_report
from sklearn.model_selection import train_test_split
from pickle import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time as tm
from Bio import SeqIO
import numpy as np
import os
import sys
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import r2_score

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Crear mapa de flujo pydot y graphviz
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
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def accuracy_m(y_true, y_pred):
    correct_predictions = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), K.floatx()))
    accuracy = correct_predictions / K.cast(K.shape(y_true)[0], K.floatx())
    return accuracy

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def squared_mean_error(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    sme = tf.reduce_mean(error)
    return sme

def r2_score(y_true, y_pred):
    sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred))
    mean_y_true = tf.reduce_mean(y_true)
    sum_squares = tf.reduce_sum(tf.square(y_true - mean_y_true))
    epsilon = tf.keras.backend.epsilon()
    sum_squares = tf.maximum(sum_squares, epsilon)
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
    # Creamos la matriz de confusión
    snn_cm = confusion_matrix(Y_validation, predictions)

    # Visualizamos la matriz de confusión
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


def create_dataset(library, MCH_out, outputDir):
    TEs = [TE for TE in SeqIO.parse(library, "fasta")]
    feature_data1 = np.zeros((len(TEs), 256, 256, 3, 4), dtype=np.int16)
    labels = np.zeros((len(TEs), 2))

    n = 0
    for TE in TEs:
        print("Doing: " + TE.id)
        TE_name = TE.id.split("#")[0]
        try:
            pages = convert_from_path(MCH_out + '/te_aid/' + TE_name + '.fa.c2g.pdf')
            print(MCH_out + '/te_aid/' + TE_name + '.fa.c2g.pdf')

            for page in pages:
                page.save(MCH_out + '/te_aid/' + TE_name + '.fa.c2g.jpeg', 'JPEG')

            te_aid_image = cv2.imread(MCH_out + '/te_aid/' + TE_name + '.fa.c2g.jpeg')

            divergence_plot = cv2.resize(te_aid_image[150:1030, 150:1130, :], (256, 256))
            coverage_plot = cv2.resize(te_aid_image[150:1030, 1340:2320, :], (256, 256))
            selfdot_plot = cv2.resize(te_aid_image[1340:2220, 150:1130, :], (256, 256))
            structure_plot = cv2.resize(te_aid_image[1340:2220, 1340:2320, :], (256, 256))

            feature_data1[n, :, :, :, 0] = divergence_plot
            feature_data1[n, :, :, :, 1] = coverage_plot
            feature_data1[n, :, :, :, 2] = selfdot_plot
            feature_data1[n, :, :, :, 3] = structure_plot

            start_pos = int(TE.id.split("_")[-2])
            TE_len = int(TE.id.split("_")[-1])
            labels[n, 0] = start_pos / 20000
            labels[n, 1] = (start_pos + TE_len) / 20000
            n += 1
        except Exception as ex:
            print("something wrong with " + TE_name)
            print(ex)

    np.save(outputDir + "/features_data.npy", feature_data1)
    np.save(outputDir + "/labels_data.npy", labels)

def cnn_branch(dim1, dim2, dim3, i):
    tf.keras.backend.clear_session()

    # Inputs
    inputs = tf.keras.Input(shape=(dim1, dim2, dim3), name="input_" + str(i))
    # layer 1
    layers = tf.keras.layers.Conv2D(2, (3, 3), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
    								kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
                                    bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
                                    padding="same", name="conv_" + str(i) + "_1")(inputs)
    layers = tf.keras.layers.Dropout(0.9, name="dropout_2d_" + str(i) + "_1")(layers)
    # layers = tf.keras.layers.SpatialDropout2D(0.9, name="dropout_2d_" + str(i) + "_1")(layers)
    layers = tf.keras.layers.BatchNormalization(axis=1, momentum=0.8, epsilon=0.001, scale=False,
                                                name="BN_" + str(i) + "_1")(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, name="max_pool_" + str(i) + "_1")(layers)    
    # layers = tf.keras.layers.AveragePooling2D((4, 4), strides=None, name="avg_pool_" + str(i) + "_1")(layers)

    # layer 2
    layers = tf.keras.layers.Conv2D(6, (3, 3), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
    								kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
                                    bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
                                    padding="same", name="conv_" + str(i) + "_2")(layers)
    layers = tf.keras.layers.Dropout(0.9, name="dropout_2d_" + str(i) + "_2")(layers)
    # layers = tf.keras.layers.SpatialDropout2D(0.9, name="dropout_2d_" + str(i) + "_2")(layers)
    layers = tf.keras.layers.BatchNormalization(axis=1, momentum=0.4, epsilon=0.001, scale=False,
                                                name="BN_" + str(i) + "_2")(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, name="max_pool_" + str(i) + "_2")(layers)
    # layers = tf.keras.layers.AveragePooling2D((4, 4), strides=None, name="avg_pool_" + str(i) + "_2")(layers)

    # layer 3
    layers = tf.keras.layers.Conv2D(12, (3, 3), strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1),
    								kernel_initializer='he_uniform',
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001),
                                    bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True,
                                    padding="same", name="conv_" + str(i) + "_3")(layers)
    layers = tf.keras.layers.Dropout(0.9, name="dropout_2d_" + str(i) + "_3")(layers)
    # layers = tf.keras.layers.SpatialDropout2D(0.9, name="dropout_2d_" + str(i) + "_3")(layers)
    layers = tf.keras.layers.BatchNormalization(axis=1, momentum=0.2, epsilon=0.001, scale=False,
                                                name="BN_" + str(i) + "_3")(layers)
    layers = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, name="max_pool_" + str(i) + "_3")(layers)
    # layers = tf.keras.layers.AveragePooling2D((4, 4), strides=None, name="avg_pool_" + str(i) + "_3")(layers)
    # tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, scale=False)(layers)

    # layer 4
    layers = tf.keras.layers.Flatten(name="flatten_" + str(i) + "_1")(layers)
    # layers = tf.keras.layers.Dense(64, activation="relu", name="output1_"+str(i))(layers)
    # layers = tf.keras.layers.Dense(32, activation="linear", name="output2_"+str(i))(layers)
    # layer end
    predictions = tf.keras.layers.Dense(32, activation="sigmoid", name="outputF_" + str(i))(layers)
    # model generation
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    return model


def auto_trimming(cnn_div, cnn_cov, cnn_dot, cnn_str):
    # concatenate layer
    combinedInput = tf.keras.layers.concatenate([cnn_div.output, cnn_cov.output, cnn_dot.output, cnn_str.output])
    # layer 1
    layers = tf.keras.layers.Dense(200, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001), name="dense_5_1")(combinedInput)
    layers = tf.keras.layers.Dropout(0.9, name="dropout_1d_5_3")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True,
                                                fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4,
                                                adjustment=None)(layers)
    # layer 2
    layers = tf.keras.layers.Dense(150, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=tf.keras.regularizers.l1(0.001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.01), name="dense_5_2")(layers)
    layers = tf.keras.layers.Dropout(0.6, name="dropout_1d_5_4")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True,
                                                fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4,
                                                adjustment=None)(layers)
    # layer 3
    layers = tf.keras.layers.Dense(120, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=tf.keras.regularizers.l1(0.001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.01), name="dense_5_3")(layers)
    layers = tf.keras.layers.Dropout(0.6, name="dropout_1d_5_5")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True,
                                                fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4,
                                                adjustment=None)(layers)
    # layer 4
    layers = tf.keras.layers.Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001), name="dense_5_4")(layers)
    layers = tf.keras.layers.Dropout(0.9, name="dropout_1d_5_6")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.6, epsilon=0.001, center=True, scale=False, trainable=True,
                                                fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4,
                                                adjustment=None)(layers)
    # layer end
    predictions = tf.keras.layers.Dense(2, activation="sigmoid", name="output_5")(layers)

    model = tf.keras.Model(inputs=[cnn_div.input, cnn_cov.input, cnn_dot.input, cnn_str.input], outputs=predictions)
    # optimizer
    # opt = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    # opt = tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
    # opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    # loss function
    # loss_fn = tf.keras.losses.CategoricalCrossentropy()
    # loss_fn = BinaryFocalLoss(gamma=2)
    # loss_fn = tf.keras.losses.MeanSquaredError()
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    # Compile model
    model.compile(loss="mean_absolute_error", optimizer=opt, metrics="mse")
    # model.compile(loss=loss_fn, optimizer=opt, metrics="accuracy", batch_size=64)
    return model

def custom_loss(y_true, y_pred):
     loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
     if tf.math.is_nan(loss):
        print("valor nan en funcion perdida")
        print(f"Loss: {loss.numpy()}")
        print(f"y_true: {y_true.numpy()}")
        print(f"y_pred: {y_pred.numpy()}")
        tf.stop_gradient(loss)
        raise ValueError("Loss is NaN, stopping training.")


def testing_models(model_path, scalerx_path, dataX_path, dataY_path):
    X_test = np.load(dataX_path)
    Y_test = np.load(dataY_path).astype(np.float32)
    
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    print("NaNs in X_test:", np.isnan(X_test).sum())
    print("NaNs in Y_test:", np.isnan(Y_test).sum())

    scalerX = NDStandardScaler()
    scalerX.load_model(scalerx_path, X_test)

    X1_test_scl = scalerX.transform(X_test)
    
    print("NaNs in X1_test_scl after scaling:", np.isnan(X1_test_scl).sum())

    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU(0.1),
                                                       'f1_m': f1_m})

    predictions = model.predict(
        [X1_test_scl[:, :, :, :, 0], X1_test_scl[:, :, :, :, 1], X1_test_scl[:, :, :, :, 2], X1_test_scl[:, :, :, :, 3]], verbose=0)
    
    predictions = np.nan_to_num(predictions, nan=0)
    
    print("NaNs in predictions:", np.isnan(predictions).sum())
    print("predictions shape:", predictions.shape)
    print(Y_test[:, 0])
    print(predictions[:, 0])

    r2_initial = calcular_y_graficar_r2(Y_test[:, 0], predictions[:, 0], "StartingPos")
    r2_final = calcular_y_graficar_r2(Y_test[:, 1], predictions[:, 1], "EndingPos")
    print("R2 starting position:" + str(r2_initial))
    print("R2 ending position:" + str(r2_final))


def calcular_y_graficar_r2(real, predicted, nombre):
    # Comprobación de la calidad de los datos
    print(f"Calculating R2 for {nombre}...")
    print(f"NaNs in real: {np.isnan(real).sum()}, NaNs in predicted: {np.isnan(predicted).sum()}")
    print(f"real shape: {real.shape}, predicted shape: {predicted.shape}")
    if real.shape != predicted.shape:
        raise ValueError("Shapes of real and predicted do not match.")
    r2 = r2_score(real, predicted)

    # Visualizacion
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
 
    if np.isnan(X_train).any() or np.isnan(Y_train).any() or np.isnan(X_dev).any() or np.isnan(Y_dev).any():
        print("hay valores nan")
        return None, None

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
    checkpoint_filepath = "/shared/home/sorozcoarias/coffea_genomes/Simon/auto_trimming/checkpoint/"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_mse",
        save_best_only=True,
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=10,         
        restore_best_weights=True  
    )
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
            batch_size = 32
            num_epochs = 200
            """image_size = (1, 27)  # We'll resize input images to this size
            patch_size = 1  # Size of the patches to be extract from the input images
            num_patches = 27
            projection_dim = 8
            num_heads = 16
            transformer_units = [
                projection_dim * 2,
                projection_dim,
            ]  # Size of the transformer layers
            print(transformer_units)
            transformer_layers = 4
            mlp_head_units = [2048, 1024, 512, 256]  # Size of the dense layers of the final classifier"""

            x_str = sys.argv[2]
            Y_str = sys.argv[3]
            
            # for debugging only
            # tf.config.run_functions_eagerly(True)

            x = np.load(x_str)
            x = x / 255.0
            #mean = np.mean(x, axis=0)
            #std = np.std(x, axis=0)
            #x = (x - mean) / std
            x = x.astype(np.float16)
            print(x[0, :, 0, 0, 1])
            y = np.load(Y_str)
            # y = y[:, 0]
            # print(y.shape)
            # print(y)

            # normalizar cada imagen individualmente
            # for i in range(len(x)):
               # mean = np.mean(x[i])
               # std = np.std(x[i])
               # x[i] = (x[i] - mean) / std
            # x = x.astype(np.float32) / 255.0

            # normalizacion min-max
            # from sklearn.preprocessing import MinMaxScaler
            # num_batches = len(x) // batch_size
            # scaler = MinMaxScaler()
            # for i in range(num_batches):
            #    start_idx = i * batch_size
            #    end_idx = (i + 1) * batch_size
            #    batch_x = x[start_idx:end_idx]
            #    x[start_idx:end_idx] = scaler.fit_transform(batch_x.reshape(-1, batch_x.shape[-1])).reshape(batch_x.shape)
            # x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

            # imputacion valores nulos
            # from sklearn.impute import SimpleImputer
            # imputer = SimpleImputer(strategy='mean')
            # x = imputer.fit_transform(x)

            validation_size = 0.2
            seed = 7
            X_train, X_test_dev, Y_train, Y_test_dev = train_test_split(x, y, test_size=validation_size,
                                                                        random_state=seed)
            X_dev, X_test, Y_dev, Y_test = train_test_split(X_test_dev, Y_test_dev, test_size=0.5, random_state=seed)

            scalerX = NDStandardScaler().fit(X_train)
            X_train = scalerX.transform(X_train)
            X_dev = scalerX.transform(X_dev)
            X_test = scalerX.transform(X_test)

            scalerX.save_model("scalerX")

            print("Information about the data")
            print("Training shape: X=" + str(X_train.shape) + ", Y=" + str(Y_train.shape) + ", Positive=" + str(
                Y_train[Y_train == 0].shape[0]) + " (" + str(Y_train[Y_train == 0].shape[0] / Y_train.shape[0]) + ")")
            print("Dev shape: X=" + str(X_dev.shape) + ", Y=" + str(Y_dev.shape) + ", Positive=" + str(
                Y_dev[Y_dev == 0].shape[0]) + " (" + str(Y_dev[Y_dev == 0].shape[0] / Y_dev.shape[0]) + ")")
            print("Test shape: X=" + str(X_test.shape) + ", Y=" + str(Y_test.shape) + ", Positive=" + str(
                Y_test[Y_test == 0].shape[0]) + " (" + str(Y_test[Y_test == 0].shape[0] / Y_test.shape[0]) + ")")

            # generate the NNs
            cnn_div = cnn_branch(X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)
            cnn_cov = cnn_branch(X_train.shape[1], X_train.shape[2], X_train.shape[3], 2)
            cnn_dot = cnn_branch(X_train.shape[1], X_train.shape[2], X_train.shape[3], 3)
            cnn_str = cnn_branch(X_train.shape[1], X_train.shape[2], X_train.shape[3], 4)
            model = auto_trimming(cnn_div, cnn_cov, cnn_dot, cnn_str)
            # summarize layers
            print(model.summary())
            tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

            path_log_base = os.path.dirname(os.path.realpath(__file__))
            training_set = [X_train[:, :, :, :, 0], X_train[:, :, :, :, 1], X_train[:, :, :, :, 2],
                            X_train[:, :, :, :, 3]]
            dev_set = [X_dev[:, :, :, :, 0], X_dev[:, :, :, :, 1], X_dev[:, :, :, :, 2], X_dev[:, :, :, :, 3]]
            test_set = [X_test[:, :, :, :, 0], X_test[:, :, :, :, 1], X_test[:, :, :, :, 2], X_test[:, :, :, :, 3]]

            history, checkpoints = run_experiment(model, training_set, Y_train, dev_set, Y_dev, batch_size, num_epochs)

            # outputs from each CNN branch
            #print(cnn_div.output.numpy())
            #print(cnn_cov.output.numpy())
            #print(cnn_dot.output.numpy())
            #print(cnn_str.output.numpy())
            acc = history.history['mse']
            np.save('accuracy_SENMAP_completeDB.npy', acc)
            val_acc = history.history['val_mse']
            np.save('val_accuracy_SENMAP_completeDB.npy', val_acc)
            loss = history.history['loss']
            np.save('loss_SENMAP_completeDB.npy', loss)
            val_loss = history.history['val_loss']
            np.save('val_los_SENMAP_completeDBnpy', val_loss)

            plt.plot(history.history['val_mse'])
            plt.plot(history.history['mse'])
            plt.legend(['val_mse','train_mse'], loc='upper right')
            plt.xlabel('Epoch')
            plt.ylabel('mse')
            plt.title('Epoch vs mse')
            plt.savefig('Train_Curve.png',  bbox_inches='tight',dpi=500)
            
            plt.plot(history.history['val_loss'])
            plt.plot(history.history['loss'])
            plt.legend(['val_loss','train_loss'], loc='upper right')
            plt.xlabel('Epoch')
            plt.ylabel('loss')
            plt.title('Epoch vs Loss')

            

            plt.figure()
            plt.plot(history.history['val_loss'])
            plt.plot(history.history['loss'])
            plt.legend(['val_loss','train_loss'], loc='lower right')
            plt.xlabel('Epoch')
            plt.ylabel('loss')
            plt.title('Epoch vs loss')
            plt.savefig('Train_Curve_los.png',  bbox_inches='tight',dpi=500)

            model.save('trained_model.h5')
            np.save("X_test.npy", X_test)
            np.save("Y_test.npy", Y_test)
        else:
            print("ERROR in parameters, usage: python3 NN_trainingV2.py train X Y")
            sys.exit(0)

    elif option == "test":
        if len(sys.argv) == 6:
            model_path = sys.argv[2]
            scalerx1_path = sys.argv[3]
            dataX1_path = sys.argv[4]
            dataY_path = sys.argv[5]
            testing_models(model_path, scalerx1_path, dataX1_path, dataY_path)
        else:
            print(
                "ERROR in parameters, usage: python3 SmartInspection.py test model_path scalerX1_path dataX1_path dataY_path data_TE_names_path")
            sys.exit(0)
    else:
        print("Option not found: " + option)

