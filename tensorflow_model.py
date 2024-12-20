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
model.fit(tf.expand_dims(X_train, axis=-1),y_train,epochs=100)
# model.summary()
# plot_model(model = model)
y_pred = model.predict(X_test)

#Define plotting function
def plot_prediction(train_data=X_train,
                    train_labels = y_train,
                    test_data = X_test,
                    test_labels = y_test,
                    prediction = y_pred):
    plt.figure(figsize=(10,7))
    train_data_show = plt.scatter(train_data,train_labels,c="b")
    test_data_show = plt.scatter(test_data,test_labels,c="g")
    prediction_data_show = plt.scatter(test_data,prediction,c="r")
    plt.legend((train_data_show,test_data_show,prediction_data_show),
               ('Training Data' , 'Test Data' , 'Prediction'))
    plt.show()

# plot_prediction(train_data=X_train,
#                     train_labels = y_train,
#                     test_data = X_test,
#                     test_labels = y_test,
#                     prediction = y_pred)

#Evaluation matrix
model.evaluate(X_test,y_test)
y_pred = tf.squeeze(y_pred)
mae = tf.metrics.mean_absolute_error(y_true= y_test,
                               y_pred= y_pred)
mse = tf.metrics.mean_squared_error(y_true=y_test,
                                    y_pred= y_pred)
print("MAE = ",mae)
print("MSE = ",mse)
