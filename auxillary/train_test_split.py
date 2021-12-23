from sklearn.model_selection import train_test_split

def train_test_split_(X, y,split_mode, train_until, validate_from, ):
    """
    Splits the overall dataset in train and test(validattion data)
    :param split_mode: 1: train test split automatically, 2: train test split with train until and validate from
    :param X: Independet variables (covariates)
    :param y: Dependent variable
    :return: Split of train and test data
    """
    #Train Test Split
    if split_mode ==1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 8, shuffle=False)
    if split_mode ==2:
        #train_until = 285
        #validate_from = 286
        X_train = X.loc[:train_until]
        X_test = X.loc[validate_from:]
        y_train = y.loc[:train_until]
        y_test = y.loc[validate_from:]

    return X_train, X_test, y_train, y_test