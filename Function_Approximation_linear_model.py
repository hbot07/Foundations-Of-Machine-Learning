import numpy as np


def fx(x):
    y = np.sin(x)
    return y


def generateX_fx(N):
    x = np.random.random(N)
    y = fx(x)
    return x, y


def plotdata():
    x, yd = generateX_fx(20)
    import matplotlib.pyplot as plt
    i = x.argsort()
    plt.plot(x[i], yd[i])
    plt.ylabel('f(x)')
    plt.xlabel('x')
    plt.show()


# plotdata()


def createA(x, D):
    N = x.size
    x = np.reshape(x, (N, 1))
    A = np.ones((N, 1))
    for i in range(1, D + 1):
        A = np.concatenate((A, x ** i), axis=1)
    return A


def train_x(A, Y):
    w = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(Y))
    return w


def predict_y(w, x):
    A = createA(x, w.shape[0] - 1)
    return A.dot(w)


def compute_mse(y, yd):
    return np.mean((y - yd) ** 2)


def plotModel(x, y, yd):
    import matplotlib.pyplot as plt
    i = x.argsort()
    plt.figure()
    plt.plot(x[i], y[i], 'g-o')
    plt.plot(x[i], yd[i], 'r-o')
    plt.ylabel('f(x)')
    plt.xlabel('x')
    plt.legend(['estimated', 'true'])
    plt.show()


def testModel(Ntest, w):
    x, yd = generateX_fx(Ntest)
    y = predict_y(w, x)
    mse = compute_mse(y, yd)
    return x, y, yd, mse
