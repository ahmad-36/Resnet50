# Flower Classification Using ResNet50

## Overview
This project implements a **Flower Classification Model** using a pre-trained **ResNet50** model with **TensorFlow and Keras**. The dataset contains images of different flowers, which are used to train and fine-tune a deep learning model.

## Features
- Downloads and processes the **Flower Photos dataset**.
- Uses **ResNet50** for feature extraction and classification.
- Implements **transfer learning** to improve model accuracy.
- Trains the model on TensorFlow's **image_dataset_from_directory** method.
- Visualizes training performance with accuracy and loss graphs.
- Uses the trained model for **flower classification predictions**.

## Requirements
Install the necessary dependencies:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Dataset Preparation
- The dataset is automatically downloaded from TensorFlow’s repository.
- The images are resized to **180x180 pixels**.
- The dataset is split into **training (80%)** and **validation (20%)** sets.

```python
import tensorflow as tf
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
```

## Loading and Preprocessing the Dataset
- The dataset is loaded using `image_dataset_from_directory`.
- Images are resized, batched, and shuffled for training.

```python
img_height, img_width = 180, 180
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
```

## Model Architecture
- Uses **ResNet50** as a feature extractor.
- Adds a **Flatten** layer and **Dense layers** for classification.
- **Freezes ResNet50 layers** to retain pre-trained weights.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50

resnet_model = Sequential()
pretrained_model = ResNet50(include_top=False, input_shape=(180,180,3), pooling='avg', weights='imagenet')

for layer in pretrained_model.layers:
    layer.trainable = False

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dense(5, activation='softmax'))
```

## Model Compilation and Training
- Uses **Adam optimizer** with a learning rate of `0.001`.
- Uses **categorical cross-entropy loss** for multi-class classification.
- Trains the model for **10 epochs**.

```python
from tensorflow.keras.optimizers import Adam

resnet_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10
history = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
```

## Model Evaluation
- Plots **accuracy** and **loss** curves to analyze model performance.

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'validation'])
plt.grid()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'validation'])
plt.grid()
plt.show()
```

## Making Predictions
- Loads an image, preprocesses it, and makes a prediction.

```python
import numpy as np
import cv2

image_path = str(list(data_dir.glob('roses/*'))[0])
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (img_height, img_width))
image = np.expand_dims(image_resized, axis=0)

pred = resnet_model.predict(image)
output_class = class_names[np.argmax(pred)]
print("The predicted class is", output_class)
```

## Conclusion
This project successfully fine-tunes **ResNet50** for flower classification using **TensorFlow and Keras**. The model achieves good accuracy and can be further improved by fine-tuning additional layers or using data augmentation techniques.

---
### Author
[Your Name]

