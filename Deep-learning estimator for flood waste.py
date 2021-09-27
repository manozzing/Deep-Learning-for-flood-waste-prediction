## Deep-learning estimator for flood waste

#Import packages"
!pip install --upgrade pip


from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

!pip install -q seaborn

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

pip install --upgrade tensorflow
import tensorflow as tf
from tensorflow import keras
import IPython
from tensorflow.keras import layers
from tensorflow.keras import Sequential

!pip install --upgrade keras-tuner
import kerastuner as kt

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Sequential

print(tf.__version__)
print(kt.__version__)

## set random seed
from numpy.random import seed
seed(18)
import tensorflow
tensorflow.random.set_seed(18)

#Import dataset
column_names = ['TB','HB','FB','Cropland', 'Road', 'River', 'Stream', 'Region', 'UR', 'Location', 'Disaster', 'Area', 'GRDP', 'PD', 'ARUR', 'Rhmax', 'Rdmax', 'Rtotal', 'Wind', 'UP', 'P', 'Wwsupply', 'Debris']
raw_dataset = pd.read_csv('Floodwaste2019 King.csv', names=column_names, delimiter=',', skipinitialspace=True, dtype = float) # read data set using pandas

data = raw_dataset.values
x, y = data[:, :-1], data[:, -1]
print(x.shape, y.shape)

##Preprocessing
# separate into train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)
# preprocessing - normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


#NN construction


#Modify to divert the number of layer

#NN construction with hypermodel which will be optimized later by Bayesian search
from kerastuner import HyperModel
class RegressionHyperModel_Deep(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def build(self, hp):
        model = Sequential()
        model.add(
            layers.Dense(
                units=hp.Int('units', 10, 500, 1, default=22),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'),
                input_shape=input_shape
            )
        )
        
        for i in range(hp.Int('num_layers', 1, 20)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=10, max_value=500, step=1), 
                                   activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'], default='relu'),
                                   input_shape=input_shape))
        
        model.add(
            layers.Dropout(
                hp.Float(
                    'dropout',
                    min_value=0.0,
                    max_value=0.1,
                    default=0.01,
                    step=0.005)
            )
        )
        
        model.add(layers.Dense(1))
        
        model.compile(
            optimizer='rmsprop',loss='mse',metrics=['mse']
        )
        
        return model



    
#Instantiate Model
input_shape = (x_train.shape[1],)
hypermodel_deep = RegressionHyperModel_Deep(input_shape)

#For next training
class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)

#Bayesian optimization
tuner_bo_deep = kt.tuners.bayesian.BayesianOptimization(
            hypermodel_deep,
            objective='mse',
            max_trials=50,
            seed=18,
            executions_per_trial=2,
            directory='2019trial_30_re')

#Search and set a model
tuner_bo_deep.search(x_train_scaled, y_train, epochs=10, validation_split=0.2, verbose=0)
best_bo_deep_model = tuner_bo_deep.get_best_models(num_models=1)[0]
best_bo_deep_model.evaluate(x_train_scaled, y_train)
best_bo_deep_model.evaluate(x_test_scaled, y_test)

best_bo_deep_model.summary()

#Print output
Predicted_Train = best_bo_deep_model.predict(x_train_scaled)
Predicted_Test = best_bo_deep_model.predict(x_test_scaled)
