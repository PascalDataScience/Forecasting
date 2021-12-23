import xgboost as xgb
import matplotlib.pyplot as plt


def xgboost(y_train, X_train):
    train_y = y_train
    train_X = X_train
    xgb_params = {
        'eta': 0.05,
        'max_depth': 10,
        'subsample': 1.0,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
    remain_num = 99

    fig, ax = plt.subplots(figsize=(10, 18))
    xgb.plot_importance(model, max_num_features=remain_num, height=0.8, ax=ax)
    plt.show()