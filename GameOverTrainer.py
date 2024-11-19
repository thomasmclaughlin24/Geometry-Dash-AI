import cv2
import tensorflow as tf
from Utils import ProcessImage
import numpy as np

class GameOverTrainer():
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128,activation="relu"),
            tf.keras.layers.Dense(2)
        ])

    def Train(self):
        labels = np.vstack((np.ones((10,1)), np.zeros((10,1))))
        n = len(labels)
        images = np.empty(shape=(n, 1024))
        for i in range(n):
            img = cv2.imread(filename="Images/GameOverImages/" + str(i) + ".png")
            if i == 0:
                img = ProcessImage(img, True, "test.png")
            else:
                img = ProcessImage(img)
            images[i] = img
        images = np.append(images, labels, axis=1)
        rng = np.random.RandomState(5)
        rng.shuffle(images)
        train, test = images[:int(n*0.8), :1024], images[int(n*0.8):, :1024]
        train_labels = images[:int(n*0.8), 1024:]
        test_labels = images[int(n*0.8):, 1024:]
        print(train[0].shape)

        self.model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        self.model.fit(train, train_labels, epochs=10)
        test_loss, test_acc = self.model.evaluate(test, test_labels, verbose=2)