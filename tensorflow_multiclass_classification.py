##
import tensorflow as tf
from matplotlib.pyplot import title
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
##
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt",
               "Sneaker","Bag","Ankle Boot"]
#Normalize the data
train_data_norm = train_data/255.0
test_data_norm = test_data/255.0
#plot multiple random images
# plt.figure(figsize=(7,7))
# for i in range(4):
#     ax=plt.subplot(2,2,i+1)
#     rand_index = random.choice(range(len(train_data)))
#     plt.imshow(train_data[rand_index],cmap=plt.cm.binary)
#     plt.title(class_names[train_labels[rand_index]])
#     plt.axis(False)
# plt.show()

##
model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),#724=28*28 Data need to be flatten
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax)
])
model2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
               optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
               metrics=["accuracy"])
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch:1e-3 * 10**(epoch/20))

norm_history = model2.fit(train_data_norm,train_labels,epochs=20,
                          validation_data=(test_data_norm,test_labels),
                          callbacks=[lr_scheduler])
y_probability = model2.predict(test_data_norm)
#convert all the predictions into int
y_pred = y_probability.argmax(axis=1)
##
#plot non-normalized data loss curve
# pd.DataFrame(non_norm_history.history).plot(title="non_norm")
# pd.DataFrame(norm_history.history).plot(title="norm")
#
#plot the learning rate decay curve # best is 1e-3
# lrs = 1e-3*(10**(tf.range(20)/20))
# plt.semilogx(lrs,norm_history.history["loss"])
# plt.xlabel("Lr")
# plt.ylabel("Loss")
# plt.title("ideal Lr")
# plt.show()

##
#create beautiful confusion matrix without tensorflow
def make_confusion_matrix(y_true,y_pred, classes =None, figsize=(10,10), text_size=15):
    cm = confusion_matrix(y_true,y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]#normalzie the confusion matrix
    n_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm,cmap=plt.cm.Blues)
    fig.colorbar(cax)
    # Set labels to be classes
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
    #Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)
    #set the color threshold
    threshold = (cm.max()+cm.min())/2.
    #plot the text
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size = text_size)
    plt.show()
##
make_confusion_matrix(y_true=test_labels,
                      y_pred=y_pred,
                      classes=class_names,
                      figsize=(15,15),
                      text_size=10)
## #Plot random image with predictions and labels

def plot_random_image(model,images,true_labels,classes):
    #set up random integer
    i = random.randint(0,len(images))
    #craete prediction and taragets
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1,28,28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]
    #plot the image
    plt.imshow(target_image,cmap=plt.cm.binary)
    #change the color of titles if the prediction is wrong
    if pred_label == true_label:
        color = "green"
    else:
        color="red"
    #Add xlabel info
    plt.xlabel("Pred: {} {:2.0f}% (True:{})".format(pred_label,
                                                    100*tf.reduce_max(pred_probs),
                                                    true_label),
               color = color)
    plt.show()

plot_random_image(model=model2,
                  images=test_data_norm,
                  true_labels=test_labels,
                  classes=class_names)