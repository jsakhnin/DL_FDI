import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn import metrics, svm
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

###################################################################################### 
#########################                Models            ###########################  
def DLmodel1(f):
    model = keras.Sequential([
       tf.keras.Input(shape=(f,)),
       tf.keras.layers.Dense(128,activation='relu'),
       tf.keras.layers.Dense(64,activation='relu'),
       tf.keras.layers.Dense(32,activation='relu'),
       tf.keras.layers.Dense(16,activation='relu'),       
       tf.keras.layers.Dense(1,activation='sigmoid'),
    ])



    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.SGD(lr=0.0001),
                 metrics=['accuracy'])
    return model 


def DLmodel2(f):
    model = keras.Sequential([
       tf.keras.Input(shape=(f,)), 
       tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),        
       tf.keras.layers.Dense(64,activation='relu'),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(32,activation='relu'),
       tf.keras.layers.Dense(16,activation='relu'),
       tf.keras.layers.Dense(1,activation='sigmoid'),      
    ])



    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                 metrics=['accuracy'])
    return model 


def DLmodel3(f):
    model = keras.Sequential([
       tf.keras.Input(shape=(f,)), 
       tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(64,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(32,activation='relu'),
       tf.keras.layers.Dense(16,activation='relu'),
       tf.keras.layers.Dense(1,activation='sigmoid'),      
    ])



    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                 metrics=['accuracy'])
    return model 


def DLmodel4(f):
    model = keras.Sequential([
       tf.keras.Input(shape=(f,)), 
       tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(64,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(32,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dense(16,activation='relu'),
       tf.keras.layers.Dense(1,activation='sigmoid'),      
    ])



    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                 metrics=['accuracy'])
    return model 

def DLmodel5(f):
    model = keras.Sequential([
       tf.keras.Input(shape=(f,)), 
       tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(64,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(32,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dense(16,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dense(1,activation='sigmoid'),      
    ])



    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                 metrics=['accuracy'])
    return model 


def DLmodel6(f):
    model = keras.Sequential([
       tf.keras.Input(shape=(f,)), 
       tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(64,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(32,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(16,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dense(1,activation='sigmoid'),      
    ])



    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                 metrics=['accuracy'])
    return model 

def DLmodel7(f):
    model = keras.Sequential([
       tf.keras.Input(shape=(f,)), 
       tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.4),
       tf.keras.layers.Dense(64,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.4),
       tf.keras.layers.Dense(32,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(16,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dense(1,activation='sigmoid'),      
    ])



    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                 metrics=['accuracy'])
    return model 


def DLmodel8(f):
    model = keras.Sequential([
       tf.keras.Input(shape=(f,)), 
       #tf.keras.layers.Dense(128,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       #tf.keras.layers.Dropout(0.4),
       #tf.keras.layers.Dense(64,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       #tf.keras.layers.Dropout(0.4),
       tf.keras.layers.Dense(32,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(16,activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
       tf.keras.layers.Dense(1,activation='sigmoid'),      
    ])



    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
                 metrics=['accuracy'])
    return model 