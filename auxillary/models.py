import pandas as pd
import numpy as np
from sklearn import linear_model
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

def linear_model_(X_train, y_train, X_test):
    """
    Trains a linear Regression Model and predicts Values for the specified Timestamps.
    """
    # Initialize Model
    lr_model = linear_model.LinearRegression()

    # Train Linear Regression Model
    lr_model.fit(X_train, y_train)

    # Predict Datapoints
    y_pred = lr_model.predict(X_test)

    return y_pred


def support_vector_machines(X_train, y_train, X_test):
    """
    Support vector machines
    """
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def linear_model_lasso(X_train,y_train, X_test):
    """
    Trains a linear Regression Model and predicts Values for the specified Timestamps.
    """
    # Initialize Model
    #lr_model = linear_model.Lasso(alpha=0.1)
    lr_model = linear_model.Lasso(alpha=0.5)


    # Train Linear Regression Model
    lr_model.fit(X_train, y_train)

    # Predict Datapoints
    y_pred = lr_model.predict(X_test)

    return y_pred


def random_forest_regressor(X_train,y_train, X_test, max_depth, random_state):
    """
    Random Forest Regressor
    """
    regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    return y_pred
"""
def neural_network(X_train, y_train, X_test, n_epochs):

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize Model
    model = Sequential()
    model.add(Dense(4, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # Train Linear Regression Model
    model.fit(X_train, y_train, epochs=n_epochs)

    # Predict Datapoints
    y_pred_arr = model.predict(X_test)
    df_pred = pd.DataFrame(y_pred_arr, columns=['y_pred'])
    y_pred = df_pred['y_pred'].values
    return y_pred
    """
if __name__ == "__main__":
    #y_pred = linear_model_(X_train, y_train, X_test)
    print("hi")