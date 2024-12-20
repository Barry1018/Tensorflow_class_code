import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import io
import requests
from flatbuffers.packer import float64
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder


url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv"
s = requests.get(url).content
insurance = pd.read_csv(io.StringIO(s.decode('utf-8')))
#Normalizing the data
ct = make_column_transformer(
    (MinMaxScaler(),["age","bmi","children"]),
    (OneHotEncoder(handle_unknown="ignore"),["sex","smoker","region"])
)


#Create data and labels(X & y)
X = insurance.drop("charges", axis=1) # everything in the chart without charges
y = insurance["charges"]
#X = np.asarray(X).astype(np.float32)# Because of bool in the chart


#Create training and testing data using sklearn train_test_split!!!!
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#Fit the column transformer to our training data
ct.fit(X_train)
# #Transform training and test data with normalization
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

#Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.legacy.Adam(),
              metrics=["mae"])
model.fit(X_train_normal,y_train,epochs=100)
model.evaluate(X_test_normal,y_test)