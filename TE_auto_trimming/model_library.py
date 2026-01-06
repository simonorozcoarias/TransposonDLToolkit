# -*- coding: utf-8 -*-
import os
from typing import List, Tuple
from pickle import dump, load

# For data and plotting 
import matplotlib.pyplot as plt
import numpy as np

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.base import TransformerMixin

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout

# GPU config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "default"
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# To create flow diagrams using pydot and graphviz
try:
    import pydot
except ImportError:
    print("pydot is not installed. Installing...")
    os.system('pip install pydot')
    import pydot

try:
    import graphviz
except ImportError:
    print("graphviz is not installed. Installing...")
    os.system('pip install graphviz')
    import graphviz

# This class flattens N-dimensional data to 2D before applying scikit-learn's StandardScaler, 
# and then reshapes the data back to its original dimensions after transformation.
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

def r2_score(y_true, y_pred):
    # Sum of squared residuals
    sum_squares_residuals = tf.reduce_sum(tf.square(y_true - y_pred))

    # Mean of the true values
    mean_y_true = tf.reduce_mean(y_true)

    # Total sum of squares
    sum_squares = tf.reduce_sum(tf.square(y_true - mean_y_true))

    # Avoid division by zero
    epsilon = tf.keras.backend.epsilon()
    
    sum_squares = tf.maximum(sum_squares, epsilon)

    # Coefficient of determination (R2)
    R2 = 1 - sum_squares_residuals / sum_squares
    return R2

# Implementation of a ResNet18-like convolutional neural network
class ResNet18:

    tf.keras.backend.clear_session()

    # Identity residual block
    @staticmethod
    def identity_block(X, filters: List[int]):
        # unpack number of filters to be used for each conv layer
        f1, f2 = filters
        X_shortcut = X

         # first convolutional layer (plus batch norm & relu activation)
        X = Conv2D(f1, (5,5), strides=(1,1), padding='same',
                   kernel_initializer='he_uniform', activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                   bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X)
        X = BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, scale=False)(X)

        # second convolutional layer
        X = Conv2D(f2, (5,5), strides=(1,1), padding='same',
                   kernel_initializer='he_uniform', activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                   bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X)
        X = BatchNormalization(axis=-1, momentum=0.4, epsilon=0.001, scale=False)(X)

        # add shortcut branch to main path
        X = Add()([X, X_shortcut])
        
        # relu activation at the end of the block
        X = Activation(tf.keras.layers.LeakyReLU(alpha=0.1))(X)

        return X

    # Convolutional residual block
    @staticmethod
    def convolutional_block(X, filters: List[int], strides: Tuple[int,int]=(2,2)):
        
        # unpack number of filters to be used for each conv layer
        f1, f2 = filters
        
        # the shortcut branch of the convolutional block
        X_shortcut = X

        # first convolutional layer
        X = Conv2D(f1, (5,5), strides=strides, padding='same',
                   kernel_initializer='he_uniform', activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                   bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X)
        X = BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, scale=False)(X)

        # second convolutional layer
        X = Conv2D(f2, (5,5), strides=(1,1), padding='same',
                   kernel_initializer='he_uniform', activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                   bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X)
        X = BatchNormalization(axis=-1, momentum=0.4, epsilon=0.001, scale=False)(X)

        # shortcut path
        X_shortcut = Conv2D(f2, (1,1), strides=strides, padding='same',
                            kernel_initializer='he_uniform', activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                            bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X_shortcut)                         
        X_shortcut = BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, scale=False)(X_shortcut)

        # add shortcut branch to main path
        X = Add()([X, X_shortcut])
        
        # nonlinearity
        X = Activation(tf.keras.layers.LeakyReLU(alpha=0.1))(X)

        return X

    # Build the full ResNet18-like model
    @staticmethod 
    def build(input_size: Tuple[int,int,int], classes: int) -> Model:
    
        # tensor placeholder for the model's input
        X_input = Input(input_size)

        # convolutional layer, followed by batch normalization and relu activation
        X = Conv2D(16, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform',
                   activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                   bias_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.001), use_bias=True)(X_input)
        X = BatchNormalization(axis=-1, momentum=0.8, epsilon=0.001, scale=False)(X)
        X = Activation(tf.keras.layers.LeakyReLU(alpha=0.1))(X)
        
        # max pooling layer to halve the size coming from the previous layer
        X = MaxPooling2D((4, 4), strides=(2,2), padding='same')(X)

        # Blocks
        X = ResNet18.convolutional_block(X, [16,16], strides=(1,1))
        X = ResNet18.identity_block(X, [16,16])

        X = ResNet18.convolutional_block(X, [32,32])
        X = ResNet18.identity_block(X, [32,32])

        X = ResNet18.convolutional_block(X, [64,64])
        X = ResNet18.identity_block(X, [64,64])

        X = ResNet18.convolutional_block(X, [128,128])
        X = ResNet18.identity_block(X, [128,128])

        # Pooling layers
        X = MaxPooling2D((4, 4), padding='same')(X)
        
        # Convert feature maps ? vector
        X = GlobalAveragePooling2D()(X)
        
        # Output layer
        X = Dense(classes, activation='sigmoid')(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet18_5x5')
        return model

# Create the model by combining the outputs of the four CNN branches 
def auto_trimming(cnn_div, cnn_cov, cnn_dot, cnn_str):
    combinedInput = tf.keras.layers.concatenate([cnn_div.output, cnn_cov.output, cnn_dot.output, cnn_str.output])

    # layer 1
    layers = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.1),
                                   kernel_regularizer=tf.keras.regularizers.l1(0.0001),
                                   kernel_initializer='he_normal', bias_regularizer=tf.keras.regularizers.l2(0.001),
                                   name="dense_5_1")(combinedInput)
    layers = tf.keras.layers.Dropout(0.2, name="dropout_1d_5_1")(layers)   
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

def plot_r2(real, predicted, name):

    # Verify real and predictions data shapes match
    if real.shape != predicted.shape:
        raise ValueError("Shapes of real and predicted do not match.")
    
    # Calculate determination coefficient
    r2 = r2_score(real, predicted)

    # Plots
    plt.figure(figsize=(8, 6))
    plt.scatter(real, predicted, color='blue', label='Datos')
    plt.plot(real, real, color='red', label=f'Referencia (y=x, R^2 = {r2:.2f})')
    plt.xlabel('Real')
    plt.ylabel('Predicted')
    plt.title('Grafico de dispersion con R^2 - ' + name)
    plt.legend()
    plt.grid(True)
    plt.savefig('r2_' + name + '.png', bbox_inches='tight', dpi=500)

    return r2

# Plots Loss function vs Epochs, and R2 vs Epochs
def plot_training_metrics(history):

    plt.figure()
    plt.plot(history.history['r2_score'])
    plt.plot(history.history['val_r2_score'])
    plt.legend(['train_r2_score', 'val_r2_score'], loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.title('Epoch vs R2 Score')
    plt.savefig('Train_Curve_R2.png', dpi=500, bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train_loss', 'val_loss'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Epoch vs Loss')
    plt.savefig('Train_Curve_Loss.png', bbox_inches='tight', dpi=500)
    plt.close()  

def test_model(model_path, scalerx_path, data_path):

    import pandas as pd
    import re
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Load X_test (tuple of 4 arrays) and Y_test
    X_test = np.load(os.path.join(data_path, "features_data.npy"))
    X_test = X_test / 255.0
    X_test = X_test.astype(np.float32)    
    
    Y_test = np.load(os.path.join(data_path, "labels_data.npy")).astype(np.float32)
    case_labels = np.load(os.path.join(data_path, "case_labels.npy"))
    
    sample_names = np.arange(X_test.shape[0])
    
    unique_cases = ["Caso1", "Caso2", "Caso3", "Caso4"]
    metrics = {}

    sample_names = np.arange(X_test.shape[0])
    output_dir = "./results_cases"
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine channels in an array (N, H, W, 4)
    #X_test = np.concatenate(X_test_tuple, axis=-1)  
    
    print("X_test raw:", np.min(X_test), np.max(X_test))
    print("Y_test raw:", np.min(Y_test), np.max(Y_test))
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)
    print("NaNs in X_test:", np.isnan(X_test).sum())
    print("NaNs in Y_test:", np.isnan(Y_test).sum())
    
    # Load scaler 
    scalerX = NDStandardScaler()
    scalerX.load_model(scalerx_path, X_test)
    X1_test_scl = scalerX.transform(X_test)
    
    print("NaNs in X1_test_scl after scaling:", np.isnan(X1_test_scl).sum())

    # Load model
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'LeakyReLU': tf.keras.layers.LeakyReLU(0.1),
            'r2_score': r2_score
        }
    )

    # Predict
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
    
    # Calculate R2
    r2_initial = plot_r2(Y_test[:, 0], predictions[:, 0], "StartingPos")
    r2_final = plot_r2(Y_test[:, 1], predictions[:, 1], "EndingPos")
    print("R2 starting position:" + str(r2_initial))
    print("R2 ending position:" + str(r2_final))

    for case in unique_cases:
    
        print(f"{case}")
        idx = np.where(np.char.startswith(case_labels.astype(str), case))[0]
        
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
    
        # Scatter plot per case 
        plt.figure(figsize=(6,6))
    
        # Output 0
        plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.6, color='blue', label='Output0')
        # Output 1
        plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.6, color='green', marker='x', label='Output1')
    
        # Reference line y=x
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
    
        # Calculate R2 for Starting and Ending positions
        r2_initial = plot_r2(Y_test[idx, 0], predictions[idx, 0], f"StartingPos_{case}")
        r2_final = plot_r2(Y_test[idx, 1], predictions[idx, 1], f"EndingPos_{case}")
    
        print(f"R2 starting position_{case}: {r2_initial:.4f}")
        print(f"R2 ending position_{case}: {r2_final:.4f}")
    
    return predictions
