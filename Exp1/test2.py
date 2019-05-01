from keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit1 = train_images[4]
digit2 = train_images[59999]

plt.imshow(digit2, cmap=plt.cm.binary)
plt.show()
