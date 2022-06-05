# -*- coding: utf-8 -*-
#pip install -q -U keras-tuner

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch

drugpairs = pd.read_csv("Neuron_input.csv")
drugpairs.head()

drugpairs.columns.values

#replace output 86 to 0 because the DDN take 86 output started from 0 and end at 85
drugpairs['Y'] = drugpairs['Y'].replace(86,0)
drugpairs['Y']

features = drugpairs.drop(['Drug1_ID', 'Drug1', 'Drug2_ID', 'Drug2', 'Y'], axis=1)
X= np.array(features)
y = drugpairs.Y

X=tf.keras.utils.normalize(X,axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = keras.models.Sequential([
  keras.layers.Dense(100,input_shape=(100,),activation='relu'),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(100,activation='tanh'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(100,activation='tanh'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(86,activation='softmax')
])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

model.evaluate(X_test,y_test)


def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers',2,30)):
        model.add(layers.Dense(units=hp.Int('units_'+str(i),
                                          min_value=10,
                                          max_value=500,
                                          step=1),
                              activation=hp.Choice('act_' + str(i), ['relu', 'tanh'])))
                        
    model.add(layers.Dense(86,activation='softmax'))
    model.compile(
        optimizer= keras.optimizers.Adam(
            hp.Choice('learning_rate',[1e-2,1e-3,1e-4])),
                     loss ='sparse_categorical_crossentropy',
                    metrics = ['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='Drug-drugfood interaction',
    project_name='DDI')

tuner.search_space_summary()

tuner.search(X_train, y_train,
            epochs=5,
            validation_data=(X_test,y_test))

tuner.results_summary()

"""# Option 2 - use only relu at the dense layer, and reduce neuron from 10-100"""

def build_model2(hp):
    model = keras.Sequential()
    model.add(layers.Dense(100,
                           input_shape=(100,),
                           activation='relu'))
              
    for i in range(hp.Int('num_layers',2,30)):
        model.add(layers.Dense(units=hp.Int('units_'+str(i),
                                          min_value=10,
                                          max_value=100,
                                          step=1),
                              activation='relu'))
                        
    model.add(layers.Dense(86,activation=hp.Choice('act_' + str(i), ['sigmoid', 'softmax'])))
    model.compile(
        optimizer= keras.optimizers.Adam(
            hp.Choice('learning_rate',[1e-2,1e-3,1e-4])),
                     loss ='sparse_categorical_crossentropy',
                    metrics = ['accuracy'])
    return model

tuner = RandomSearch(
    build_model2,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='Drug-drugfood interaction',
    project_name='DDI2')

tuner.search(X_train, y_train,
            epochs=5,
            validation_data=(X_test,y_test))

tuner.results_summary()

"""# Option 3 use relu/tanh at the dense layer, use softmax at the last layer and reduce neuron from 86-100"""

def build_model3(hp):
    model = keras.Sequential()
    model.add(layers.Dense(100,
                           input_shape=(100,),
                           activation='relu'))
              
    for i in range(hp.Int('num_layers',2,10)):
        model.add(layers.Dense(units=hp.Int('units_'+str(i),
                                          min_value=86,
                                          max_value=100,
                                          step=1),
                              activation=hp.Choice('act_' + str(i), ['relu', 'tanh'])))
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(86,activation='softmax'))
    model.compile(
        optimizer= keras.optimizers.Adam(
            hp.Choice('learning_rate',[1e-2,1e-3,1e-4])),
                     loss ='sparse_categorical_crossentropy',
                    metrics = ['accuracy'])
    return model

tuner = RandomSearch(
    build_model3,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='Drug-drugfood interaction',
    project_name='DDI3')

tuner.search(X_train, y_train,
            epochs=5,
            validation_data=(X_test,y_test))

tuner.results_summary()