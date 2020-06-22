from sklearn.datasets import load_iris
import numpy as np
import keras

# print("Reached here")
# np.random.seed(0)
iris = load_iris()
X = iris['data']
Y = iris['target']
# print(X.shape, Y.shape)
# print(iris.keys())
# print(iris['data'][:3])
# print(iris['target'][:3])

from keras.utils import to_categorical

Ny = len(np.unique(Y))
print(Ny)
Y = to_categorical(Y[:], num_classes=Ny)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
print('X_train shape:', X_train.shape)
print('X_test.shape:', X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# print(X_test)
addlcol = lambda x: np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
Ns, Nx = X_train.shape
print('Ns: ', Ns, 'Nx: ', Nx)


def find_weights(A, Y):
    print(A.shape)

    print(Y.shape)

    weights = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(Y))
    return weights


A = addlcol(X_train)
Y = Y_train
w = find_weights(A, Y)
# print(w)
B = addlcol(X_test)


# print(B[1].T.dot(w), Y_test[1])

def evaluate(X, W, Yd, transform_X_a):
    a = transform_X_a(X)
    yd = np.argmax(Yd, axis=1)
    y = np.argmax(a.dot(W), axis=1)
    print('Confusion Matrix:')
    print(confusion_matrix(yd, y))


evaluate(X_train, w, Y_train, addlcol)
evaluate(X_test, w, Y_test, addlcol)

addSqlcol = lambda x: np.concatenate((x, x ** 2, np.ones((x.shape[0], 1))), axis=1)

A = addSqlcol(X_train)
Y = Y_train
w = find_weights(A, Y)
evaluate(X_train, w, Y_train, addSqlcol)
evaluate(X_test, w, Y_test, addSqlcol)
