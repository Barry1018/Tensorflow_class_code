import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

#Evaluating
X_new = tf.range(-100, 100, 4)
y_new = X_new + 10

X_train = X_new[:40]
X_test = X_new[40:]

y_train = y_new[:40]
y_test = y_new[40:]

# plt.figure(figsize=(10,7))
# plt.scatter(X_train,y_train,c='b',label="Training Data")
# plt.scatter(X_test,y_test,c='g',label="Test Data")
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1]),
    tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.legacy.SGD(),
              metrics=["mae"])
model.fit(tf.expand_dims(X_train, axis=-1),y_train,epochs=50)
# model.summary()
# plot_model(model = model)
y_pred = model.predict(X_test)

#Save model
model.save("model_test_save")
model.save("model_test_HDF5.h5")