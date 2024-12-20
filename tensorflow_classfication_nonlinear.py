from sklearn.datasets import make_circles
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow_model import X_train, y_train

#Make 1000 samples
n_samples = 1000
#Create circles
X,y=make_circles(n_samples,
                 noise=0.03,
                 random_state=42)
circles = pd.DataFrame({"X0":X[:,0], "X1":X[:,1], "label":y})
X_train , y_train = X[:800] , y[:800]
X_test , y_test = X[800:] , y[800:]
#Create non-linear Model
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(40,activation="relu"),
    tf.keras.layers.Dense(40),
    tf.keras.layers.Dense(1,activation="softmax")
]
)

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
              metrics=["accuracy"])
#Create a learning rate callback
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch/20))
history = model.fit(X_train,y_train,epochs=50,callbacks=[lr_callback])
model.evaluate(X_test,y_test)
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:,1].max() + 0.1
    xx , yy = np.meshgrid(np.linspace(x_min,x_max,100),
                          np.linspace(y_min,y_max,100))
    x_in = np.c_[xx.ravel(),yy.ravel()]#Stack 2D arrays together
    y_pred = model.predict(x_in)
    #Check for multi-class

    if len(y_pred[0])>1:
        print("Doing multi-class classification")
        y_pred = np.argmax(y_pred,axis=1).reshape(xx.shape)
    else:
        print("Doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)
    plt.contourf(xx,yy,y_pred,cmap = plt.cm.RdYlBu, alpha = 0.7)
    plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    # plt.show()

# plot_decision_boundary(model,X,y)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model,X_test,y_test)


#Convert the history object into a DataFrame
pd.DataFrame(history.history).plot()
plt.show()