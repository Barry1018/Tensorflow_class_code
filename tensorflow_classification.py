from scipy.stats import alpha
from sklearn.datasets import make_circles
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#Make 1000 samples
n_samples = 1000
#Create circles
X,y=make_circles(n_samples,
                 noise=0.03,
                 random_state=42)
circles = pd.DataFrame({"X0":X[:,0], "X1":X[:,1], "label":y})
print(circles)
# plt.scatter(X[:,0],X[:,1],cmap = plt.cm.RdYlBu, alpha = 0.7)
# plt.show()

#Create Model
tf.random.set_seed(42)
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1, activation="relu"),
])

model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.legacy.Adam(lr=0.01),
                metrics=["accuracy"])

model_1.fit(X,y,epochs=5)
model_1.evaluate(X,y)
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
    plt.show()

plot_decision_boundary(model_1,X,y)