import tensorflow as tf
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ds=pd.read_csv("/dataset.csv")
print(ds)
ds=pd.read_csv("/dataset.csv")
print(ds)

features = data[['size','rows']].values
target = data['amount'].values
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train_max =y_train.max()
y_test =y_test/y_train_max
y_train =y_train/y_train_max

#define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape(1),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])