import numpy as np
def softmax(z): #z is a vector
    E = np.exp(z)
    P = E / np.sum(E, axis=0, keepdims=True)
    return P