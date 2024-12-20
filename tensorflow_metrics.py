from sklearn.datasets import make_circles
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
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
loss, accuracy = model.evaluate(X_test,y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

#Create a confusion matrix
y_pred = model.predict(X_test)
print(confusion_matrix(y_test,np.round(y_pred))) #Because y_pred is in prediction probability form

#create beautiful confusion matrix without tensorflow
figsize = (10,10)
cm = confusion_matrix(y_test,np.round(y_pred))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]#normalzie the confusion matrix
n_classes = cm.shape[0]
fig, ax = plt.subplots(figsize=figsize)
cax = ax.matshow(cm,cmap=plt.cm.Blues)
fig.colorbar(cax)
# create classes
classes = False
if classes:
    labels = classes
else:
    labels = np.arange(cm.shape[0])

#label the axes
ax.set(title="Confusion Matrix",
       xlabel="Predicted label",
       ylabel="True label",
       xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       yticklabels=labels,
       xticklabels=labels)
#set the color threshold
threshold = (cm.max()+cm.min())/2.
#plot the text
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black")
plt.show()