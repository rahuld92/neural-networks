import numpy as np
from sklearn.neural_network import MLPClassifier
#from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
import pandas as pd
import matplotlib.pyplot as plt


def get_data():
    df = pd.read_csv("data.csv")
    training_data = df.to_numpy()
    X = training_data[:, 0:-2]
    X = normalize(X, axis=0)
    X = np.hstack((X, training_data[:, 3:-1]))
    n, m = X.shape
    X = np.hstack((np.ones((n, 1)), X))
    Y = np.array(training_data[:, -1])
    return X, Y


x_train, y_train = get_data()

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                   hidden_layer_sizes=(5,1), random_state=1, learning_rate_init=0.1, max_iter=10000,
                   n_iter_no_change=300, activation="tanh")
clf.fit(x_train, y_train)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(0.1))
plt.plot(clf.loss_curve_)
plt.show()
print(clf.loss_)
