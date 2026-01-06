import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, SpatialDropout2D, AveragePooling2D, BatchNormalization, LeakyReLU, Input

def recall_m(y_true, y_pred):
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(y_pred, K.floatx())
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(y_pred, K.floatx())
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_senmap_model(input_shape=(5, 23200, 1), num_classes=3):
    """
    Returns the SENMAP model architecture.
    Adapted for 3 classes: 0 (Removed), 1 (Kept), 2 (No_TE)
    """
    tf.keras.backend.clear_session()

    # Inputs
    inputs = Input(shape=input_shape, name="input_1")
    
    # layer 1
    layers = Conv2D(32, (5, 51), strides=(1, 1), activation=LeakyReLU(0.01), 
                    kernel_regularizer=regularizers.l1_l2(0.0000, 0.000), 
                    bias_regularizer=regularizers.l1_l2(0.0000, 0.0000), use_bias=True)(inputs)
    layers = SpatialDropout2D(0.2)(layers)
    layers = AveragePooling2D((1, 9), strides=None)(layers)
    layers = BatchNormalization(axis=-1, momentum=0.6, epsilon=0.001, scale=False)(layers)

    # layer 2
    layers = Conv2D(64, (1, 31), strides=(1, 1), activation=LeakyReLU(0.01), 
                    kernel_regularizer=regularizers.l1_l2(0.0000, 0.000), 
                    bias_regularizer=regularizers.l1_l2(0.0000, 0.0000), use_bias=True)(layers)
    layers = SpatialDropout2D(0.2)(layers)
    layers = AveragePooling2D((1, 9), strides=None)(layers)
    layers = BatchNormalization(axis=1, momentum=0.6, epsilon=0.001, scale=False)(layers)

    # layer 3
    layers = Conv2D(128, (1, 11), strides=(1, 1), activation=LeakyReLU(0.01), 
                    kernel_regularizer=regularizers.l1_l2(0.0000, 0.000), 
                    bias_regularizer=regularizers.l1_l2(0.0000, 0.0000), use_bias=True)(layers)
    layers = SpatialDropout2D(0.2)(layers)
    layers = AveragePooling2D((1, 7), strides=None)(layers)
    layers = BatchNormalization(axis=1, momentum=0.6, epsilon=0.001, scale=False)(layers)
    
    # layer 4
    layers = Conv2D(256, (1, 5), strides=(1, 1), activation=LeakyReLU(0.01), 
                    kernel_regularizer=regularizers.l1_l2(0.0000, 0.000), 
                    bias_regularizer=regularizers.l1_l2(0.0000, 0.0000), use_bias=True)(layers)
    layers = SpatialDropout2D(0.2)(layers)
    layers = AveragePooling2D((1, 5), strides=None)(layers)
    layers = BatchNormalization(axis=1, momentum=0.6, epsilon=0.001, scale=False)(layers)

    # layer 5
    layers = Flatten()(layers)

    # layer 6
    layers = Dense(300, activation=LeakyReLU(0.01), 
                   kernel_regularizer=regularizers.l1_l2(0.0003, 0.001), 
                   bias_regularizer=regularizers.l1(0.001))(layers)
    layers = Dropout(0.2)(layers)
    layers = BatchNormalization(momentum=0.6, epsilon=0.001, scale=False)(layers)

    # layer 7
    layers = Dense(300, activation=LeakyReLU(0.01), 
                   kernel_regularizer=regularizers.l1_l2(0.0003, 0.001), 
                   bias_regularizer=regularizers.l1(0.001))(layers)
    layers = Dropout(0.2)(layers)
    layers = BatchNormalization(momentum=0.6, epsilon=0.001, scale=False)(layers)

    # layer 8
    layers = Dense(300, activation=LeakyReLU(0.01), 
                   kernel_regularizer=regularizers.l1_l2(0.0003, 0.001), 
                   bias_regularizer=regularizers.l1(0.001))(layers)
    layers = Dropout(0.2)(layers)
    layers = BatchNormalization(momentum=0.6, epsilon=0.001, scale=False)(layers)

    # layer end
    # Changed to num_classes (default 3)
    predictions = Dense(num_classes, activation="softmax", name="output_1")(layers)
    
    # model generation
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    
    return model
