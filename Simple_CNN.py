####################
## Input handling ##
####################

class DataPrep:
    def __init__(self):
        pass

    @staticmethod
    def dataPrep(in_data):
        if len(in_data.Y.shape) == 2:
            sel = in_data.Y[:, 1]
            X_train = in_data.X[sel == 0, :]
            X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1],1])
            X_test  = in_data.X[sel == 1, :]
            X_test  = np.reshape(X_test, [X_test.shape[0], X_test.shape[1],1])
            Y_train = in_data.Y[sel == 0, 0]
            Y_test  = in_data.Y[sel == 1, 0]
        else:
            X_train = in_data.X
            X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1],1])
            Y_train = in_data.Y[:, 0]
            X_test  = None
            Y_test  = None
        return (X_train, Y_train, X_test, Y_test)

##########################################
## Simple Convolutional Neural Networks ##
##########################################

from tensorflow.keras import models, optimizers, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from Orange.data import Table
import numpy as np

class SimpleRegressor:
    def __init__(self):
        pass

    @staticmethod
    def build(spectrumLength, activation="relu", optimizer="adam"):
        """
        Parameters
        -----------
        spectrumLength : Number of features in the spectra

        Returns
        -------
        model : The model
        """
        input_layer=layers.Input(shape=(spectrumLength, 1))
        x = layers.Conv1D(8, kernel_size=11, padding='valid', strides=1)(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation=activation)(x)
        out = layers.Dense(1, activation='linear')(x)
        
        model = models.Model(input_layer, out)
        opt = "adam" if optimizer is None else optimizer
        model.compile(loss='mae', optimizer=opt, metrics=['root_mean_squared_error'])
        return model

class SimpleClassifier:
    def __init__(self):
        pass
    
    @staticmethod
    def build(spectrumLength, activation="relu", num_classes=2, optimizer=None):
        """
        Parameters
        -----------
        spectrumLength : Number of features in the spectra

        Returns
        -------
        model : The model
        """
        input_layer=layers.Input(shape=(spectrumLength, 1))
        x = layers.Conv1D(8, kernel_size=11, padding='valid', strides=1)(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation=activation)(x)
        if num_classes == 2:
            out = layers.Dense(1, activation='sigmoid')(x)
            model = models.Model(input_layer, out)
            model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-03), metrics=['accuracy'])
        else:
            out = layers.Dense(num_classes, activation='softmax')(x)
            model = models.Model(input_layer, out)
            opt = "adam" if optimizer is None else optimizer
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

###########################
## Applied to input data ##
###########################

def CNN(classify, X_train=None, Y_train=None, X_test=None, Y_test=None, batch_size=64, epochs=10, verbose=1, shuffle=True, lr=0.01):
    # Classification
    if classify:
        sc = SimpleClassifier()
        model = sc.build(X_train.shape[1], activation="relu", num_classes=len(np.unique(Y_train)), optimizer=Adam(lr=lr))
        Y_train_b = to_categorical(Y_train)
        # With test data
        if not (type(Y_test) == type(None)):
            Y_test_b = to_categorical(Y_test)
            history = model.fit(X_train, Y_train_b, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle, validation_data = (X_test, Y_test_b))
        # Only training data
        else:
            history = model.fit(X_train, Y_train_b, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    
    # Regression
    else:
        sr = SimpleRegressor()
        model = sr.build(X_train.shape[1], activation="relu", optimizer=Adam(lr=lr))
        if not (type(Y_test) == type(None)):
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle, validation_data = (X_test, Y_test))
        else:
            history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle)
    out_data = Table.from_numpy(None,np.hstack([np.array(history.history['accuracy'])[:,np.newaxis], np.arange(1,epochs + 1,1)[:,np.newaxis]]))
    return history
