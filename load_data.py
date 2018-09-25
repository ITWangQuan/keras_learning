# 本文件用来编写载入数据的API
import numpy as np
def load_data(path="./mnist/mnist.npz"):
    path = path
    f = np.load(path)
    X_train, y_train = f["x_train"], f["y_train"]
    X_test, y_test = f["x_test"], f["y_test"]
    f.close()
    return (X_train, y_train), (X_test, y_test)
