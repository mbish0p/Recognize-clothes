import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ---- importing data

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

# print(train_images[1])
# print(train_labels[1])

#plt.imshow(train_images[1])
#plt.show()

# ----- setting model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    # softmax - probality of outpu layer function
    keras.layers.Dense(10, activation='softmax')
])

# ----- setting loss fuction and other secifications funcions to our model

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ------ training model

# epochs how many times our network gets this same picture
model.fit(train_images, train_labels, epochs=5)

# ----- testing our model on test images and chcecking accurancy

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n Accuracy test', test_acc)


# ----- make preditions about images

predition = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel('Predicted ' + class_names[test_labels[i]])
    plt.title('Acctual ' + class_names[np.argmax(predition[i])])
    plt.show()
