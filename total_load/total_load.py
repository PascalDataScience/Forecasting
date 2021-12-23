from api_entose import load_data
from auxillary.models import linear_model_, linear_model_lasso, random_forest_regressor
from auxillary.featues import get_temporal_features_
from auxillary.train_test_split import train_test_split_
from auxillary.error_functions import MAPE, RMSE
import pandas as pd
import os
import matplotlib.pyplot as plt

from dotenv import load_dotenv

def plot_results(df_pred, y_test,  plot_title, folder):
    """
    Plot results
    """

    # X_test = X_test[np.abs(X_test)< .1]= 0

    fig = plt.figure(figsize=[15, 7.5])
    #for y_pred, name, color in zip(lst_pred, lst_names, lst_colors):
    #    plt.plot(y_pred, color=color, label=name)
    plt.plot(df_pred["Lineare Regression"], color='green', label="Lineare Regression", linewidth = 0.75)
    plt.plot(df_pred["Random Forest Regressor"], color='red', label="Random Forest Regressor", linewidth = 0.75)
    plt.plot(df_pred["Lasso Regression"], color='blue', label="Lasso Regression", linewidth = 0.75)
    plt.plot(y_test, color='black', label="Actual Total Load")
    plt.xlabel('Validation Timeseries')
    plt.ylabel('Total Load')
    plt.title(plot_title)
    plt.grid()
    #plt.ylim([200, max(y_test) * 1.3])
    #plt.ylim([5000, 22000])
    plt.legend()
    # plt.show()
    #directory = os.path.join(
    #    r"C:\Users\pascs\OneDrive\Desktop\Master_Data_Science\Customer Data Analytics\CDA_Project\Prediction_Plots", folder)
    #if not os.path.exists(directory):
    #    os.makedirs(directory)
    #fig.savefig(folder)

    return


if __name__ == "__main__":
    #Set Parameters for API-Query

    load_dotenv()
    token = os.environ['TP_API']

    start = pd.Timestamp('20180101', tz='Europe/Zurich')
    end = pd.Timestamp('20211130', tz='Europe/Zurich')
    country_code = 'CH'
    file_path = "G:\Meine Ablage\Master_Data_Science\Energy Systems and IoT\Project"
    file_name = "Total_Load_Actual.csv"

    #Load Data from Transparency Platform and put it to CSV
    df = load_data(start, end,country_code, file_path, file_name, token)

    #----------------------------------------------------------------------------------------------------------------

    #Get Data from CSV
    df = pd.read_csv(os.path.join(file_path, file_name), sep=',', index_col=0)

    #----------------------------------------------------------------------------------------------------------------
    #Create Covariates
    #tbd
    #Temperature, Wheather
    #Temporal Features

    #----------------------------------------------------------------------------------------------------------------

    #Select Timeseries as target variables and covariates
    df.index = pd.to_datetime(df.index, utc= True).tz_convert("CET")

    df_temp = pd.read_csv(os.path.join(file_path, "temperature.csv"), sep = ";", index_col=0)
    df_temp.index = pd.to_datetime(df_temp.index, utc=True).tz_convert("CET")

    #Get Intersection between temperature and actual load (some data is missing)
    lst_intersection = list(sorted(set(df.index.tolist()).intersection(df_temp.index.tolist()).intersection(df.index.tolist())))
    df_temp = df_temp.loc[lst_intersection]
    df = df.loc[df_temp.index]


    cov = get_temporal_features_(df.index, ["weekhour", "dayofyear","weekday", "hour", "holiday"], n_cos_sin_hour = 2, n_cos_sin_weekday = 2, n_cos_sin_year = 2)


    df = pd.concat([df,cov,df_temp], axis = 1)
    #---------------------------------------------------------------------------------------------------
    #Feature Importance
    import xgboost as xgb
    X =  df[df.columns.tolist()[2:]]
    #X = df[["dayofyear_sin_1", "hour_sin_1", "dayofyear_cos_2"]]
    y = df["Actual Load"]
    X_train, X_test, y_train, y_test = train_test_split_(X, y,2, train_until="2020-12-31 23:00:00+01:00",
                                                         validate_from="2021-01-01 01:00:00+01:00")


    #------------------------------------------------------------------------
    df = df.loc["2019-01-01 00:00:00+01:00":]
    covariates = [df[df.columns.tolist()[2:]],df[["OBFELDEN"]],df[df.columns.tolist()[2:-1]] ]

    covariate_names = ["Temporal Features + Temperature", "Temperature", "Temporal Features"]

    df_eval = pd.DataFrame()
    df_eval2 =pd.DataFrame()
    #-------------------------------------------------------------------------------------
    # Linear Model
    y_pred_lin = linear_model_(X_train, y_train, X_test)
    # Evaluate Predictions (MAPE)

    # Random Forest Regressor
    y_pred_RFR = random_forest_regressor(X_train, y_train, X_test, max_depth=4, random_state=0)

    # Support Vector Machines
    # y_pred_SVM = support_vector_machines(X_train, y_train, X_test)

    # Lasso Regression
    y_pred_LAREG = linear_model_lasso(X_train, y_train, X_test)

    # Neural Network Tensorflow
    # y_pred_NN = neural_network(X_train, y_train, X_test, n_epochs = 800)


    # Generate Dataframe of MAPE
    lst_pred = [y_pred_lin, y_pred_RFR, y_pred_LAREG]
    lst_pred_names = ["Lineare Regression", "Random Forest Regressor", "Lasso Regression"]
    name = "MAPE"
    lst_MAPE = [MAPE(y_test, y_pred) for y_pred in lst_pred]
    df_MAPE = pd.DataFrame(data=lst_MAPE, index=lst_pred_names, columns=[name])
    df_eval = pd.concat([df_eval, df_MAPE], axis=1)

    # Plot Results
    df_pred = pd.DataFrame(data=lst_pred, index=lst_pred_names, columns=y_test.index).transpose()
    # lst_colors = ["red", "green", "blue", "orange", "yellow", "purple", "lime", "magenta"][:len(lst_pred_names)]

    plot_results(df_pred, y_test, plot_title="Total Load Comparison Covariates: " + name,
                 folder=os.path.join(file_path, name + ".jpg"))