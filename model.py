import os
import numpy as np
from statistical_formulas import softmax

if os.path.exists("data/train.npy"):
    data = np.load("data/train.npy")
else:
    data = np.loadtxt("data/archive/mnist_train.csv", delimiter=",")
    np.save("data/train.npy", data)

labels = data[:, 0]
pixels = data[:, 1:]

X = pixels.T / 255
V = np.random.randn(128, 784) * np.sqrt(2 / 784)
W = np.random.randn(10, 128)  * np.sqrt(2 / 128)

b = np.zeros([128, 1])
c = np.zeros([10, 1])

h = np.maximum(0, V @ X + b) #intermediate variable h needed for ReLU activation
y = (W @ h + c)

Y = np.zeros([10, 60000])
Y[labels.astype(int), np.arange(60000)] = 1

P = softmax(y)
L = -np.sum(Y * np.log(P)) / 60000


print(X.shape)
print(V.shape)
print(W.shape)

print(y.shape)
print(Y.shape)

