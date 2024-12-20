import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import io
import requests
from flatbuffers.packer import float64
from sklearn.model_selection import train_test_split



url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv"
s = requests.get(url).content
insurance = pd.read_csv(io.StringIO(s.decode('utf-8')))

#One hot encoding
insurance_one_hot = pd.get_dummies(insurance)

print(insurance_one_hot.head())

#Create data and labels(X & y)
X = insurance_one_hot.drop("charges", axis=1) # everything in the chart without charges
y = insurance_one_hot["charges"]
X = np.asarray(X).astype(np.float32)# Because of bool in the chart
#Create training and testing data using sklearn train_test_split!!!!
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Create model
tf.random.set_seed(42)

insurance_model= tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.legacy.Adam(),
                        metrics=["mae"])
hsitory = insurance_model.fit(X_train,y_train,epochs=200,verbose=1)
insurance_model.evaluate(X_test,y_test)
#Plot the loss history
pd.DataFrame(hsitory.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()