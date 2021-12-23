import numpy as np

def MAPE(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

def RMSE(y_test, y_pred):
    return np.sqrt(((y_pred - y_test) ** 2).mean())
