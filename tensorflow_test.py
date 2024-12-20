from cProfile import label

import tensorflow as tf
print(tf.__version__)
devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)

#Creating Data
import numpy as np
import matplotlib.pyplot as plt
###Heello github
X = np.array([-7.0 , -4.0 , -1.0 , 2.0 , 5.0 , 8.7 , 9.5 , 6.3])
y = np.array([3.0 , 6.0 , 9 , 12 , 15 , 18.7 , 19.5 , 16.3])

# plt.scatter(X,y)
# plt.show()
#Turn numpy arrays into tensors
X = tf.cast(tf.constant(X), dtype=tf.float32)
y = tf.cast(tf.constant(y), dtype=tf.float32)

#Modeling

tf.random.set_seed(42)
#1.create the model
# model = tf.keras.Sequential([
#   tf.keras.layers.Dense(50, activation = None),
#   tf.keras.layers.Dense(1)
# ])
# #2.Compile the model
# model.compile(loss=tf.keras.losses.mae,#mean average err
#               optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=0.01),#stochasric gradient decent
#               metrics=["mae"])
# #3.Fit the model
# model.fit(tf.expand_dims(X, axis=-1), y ,epochs=100)
#
#
# #4. Predict
# y_pred = model.predict([7.0])
# print(y_pred)


# plt.show()
