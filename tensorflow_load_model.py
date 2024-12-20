import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
X_new = tf.range(-100, 100, 4)
y_new = X_new + 10

X_train = X_new[:40]
X_test = X_new[40:]

y_train = y_new[:40]
y_test = y_new[40:]

loaded_model = tf.keras.models.load_model("model_test_HDF5.h5")
loaded_model.summary()

loaded_model.predict(X_test)