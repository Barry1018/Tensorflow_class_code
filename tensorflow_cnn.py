import zipfile
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import kagglehub
import wget
import os
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Download part of the dataset Food101
# url = "https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip"
# wget.download(url)
# zip_ref = zipfile.ZipFile("pizza_steak.zip")
# zip_ref.extractall()
# zip_ref.close()

#Inspect the data
num_steak_img_train = len(os.listdir("pizza_steak/train/steak"))
print(num_steak_img_train)
#Get the class name programmatically
data_dir = pathlib.Path("pizza_steak/train")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
print(class_names)

#Visualize our images
def view_random_img(target_dir, target_class):
    target_floder = target_dir+target_class #Set up target directories
    random_image = random.sample(os.listdir(target_floder),1)
    img = mpimg.imread(target_floder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    plt.show()
    print(f"Image shape: {img.shape}" )
    return img

# img = view_random_img(target_dir="pizza_steak/train/",
#                       target_class="pizza")


#An end-to-end example
tf.random.set_seed(42)
#set up directories
train_dir = "pizza_steak/train"
test_dir = "pizza_steak/test"
#get all the pixel value between 0~1
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
#Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32,
                                               target_size=(224,224),
                                               class_mode="binary",
                                               seed =42)
valid_data = valid_datagen.flow_from_directory(directory=test_dir,
                                               batch_size=32,
                                               target_size=(224,224),
                                               class_mode="binary",
                                               seed=42)
# Build a CNN model
model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=10,
                           kernel_size=3,
                           activation="relu",
                           input_shape=(224,224,3)),
    tf.keras.layers.Conv2D(10,3,activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=2,
                              padding="valid"),
    tf.keras.layers.Conv2D(10,3,activation="relu"),
    tf.keras.layers.Conv2D(10,3,activation="relu"),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model_1.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.legacy.Adam(),
                metrics=["accuracy"])
history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))
