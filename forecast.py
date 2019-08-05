def all_paths():
    #-----------------------------ADAPT THIS DIRECTORY TO YOUR SYSTEM---------------------------------------------------
    main_branch = r"~/Desktop/CloudCrystal/predictor"
    #-----------------------------ADAPT THIS DIRECTORY TO YOUR SYSTEM---------------------------------------------------
    
    data_and_scripts_location = {
    # data paths
    "general_data_path": main_branch + r"/Data",
    "data_path": main_branch + r"/Data",
    "Data": main_branch + r"/Data",
    #raw data path
    "raw_data_path": main_branch + r"/Data/raw",
    "Data/raw": main_branch + r"/Data/raw",
    #parsed data path
    "parsed_data_path": main_branch + r"/Data/parsed",
    "Data/parsed": main_branch + r"/Data/parsed",
    #python scripts path
    "scripts_path": main_branch + r"/Lib",
    "Lib": main_branch + r"/Lib",
    #path to keras models
    "models_path": main_branch + r"/Models",
    "Models": main_branch + r"/Models",
    #path of the forecasts
    "forecasts_path": main_branch + r"/Forecasts",
    "Forecasts": main_branch + r"/Forecasts",
    # path to figures
    "Figures": main_branch + r"/Figures"
    }
    return data_and_scripts_location

def convert_path(full_path):
    #-----------------------------------------------------------------------------------------------------
    #make sure the path is OS indipendent (only for KERAS)
    from os.path import expanduser
    from os.path import join
    home = expanduser("~")
    full_path_w=full_path.split("/")
    conv_full_path = home
    for f in full_path_w[1:]:
        conv_full_path = join(conv_full_path,f)
    return conv_full_path
    #-----------------------------------------------------------------------------------------------------


def lstm_forecast(lag, model_name, forecasting_set_csv, test_model=True):

#lag = 91
#model_name = "20181005021122.h5"
#forecasting_set_csv = "test_set_size=1015_week=89.csv"
#export = True
#test = True

    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #-----------------------------------------------IMPORT TIME & DATETIME----------------------------------------------
    import time
    import datetime
    #-----------------------------------------------IMPORT TIME & DATETIME----------------------------------------------
    start_time = time.time()

    #-----------------------------------------------IMPORT PANDAS-------------------------------------------------------
    import pandas as pd
    #-----------------------------------------------IMPORT PANDAS-------------------------------------------------------

    #from allPaths import all_paths
    data_and_scripts_location = all_paths()
    models_path = data_and_scripts_location["Models"]
    data_path = data_and_scripts_location["Data/parsed"]
    forecast_path = data_and_scripts_location["Forecasts"]
    fig_path = data_and_scripts_location["Figures"]

    dataset_forecast_original = pd.read_csv(convert_path(data_path + "/" + forecasting_set_csv))
    dataset_forecast_mod = pd.read_csv(convert_path(data_path + "/" + forecasting_set_csv))


    # Forecast for every "Partition"
    #PartitionTypes = ["standard","small","highmem","K80","all"]
    PartitionTypes = ["standard","small","highmem","K80"]

    #-----------------------------------------------IMPORT COPY--------------------------------------------------------
    import copy
    #-----------------------------------------------IMPORT COPY--------------------------------------------------------

    dataset_forecast_enlarge = copy.copy(dataset_forecast_mod.iloc[:lag])
    dataset_forecast_out = copy.copy(dataset_forecast_mod.iloc[lag:-1])
    for p in PartitionTypes:
        dataset_forecast_out["Partition"] = p
        dataset_forecast_enlarge = dataset_forecast_enlarge.append(dataset_forecast_out,ignore_index=True)

    #-----------------------------------------------IMPORT NUMPY--------------------------------------------------------
    import numpy as np
    #-----------------------------------------------IMPORT NUMPY--------------------------------------------------------
        
    indexes_partitions = np.zeros((dataset_forecast_enlarge.shape[0],len(PartitionTypes)))
    i=0
    for p in PartitionTypes:
        index_partition = copy.copy(dataset_forecast_enlarge.iloc[:,0])
        index_partition.iloc[0:lag] = False
        index_partition.iloc[lag:-1] = dataset_forecast_enlarge.iloc[lag:-1,0] == p
        index_partition.iloc[-1] = dataset_forecast_enlarge.iloc[-1,0] == p
        indexes_partitions[:,i] = index_partition.values
        i += 1 

    #test=indexes_partitions==1
    #dataset_forecast_enlarge.iloc[test[:,1]]
        
    dataset_forecast = copy.copy(dataset_forecast_enlarge)
    dataset_forecast['Partition'] = dataset_forecast['Partition'].astype('category')
    dataset_forecast['Partition'] = dataset_forecast['Partition'].cat.codes


    def qt_to_class(queueTimes):
        queueTimeAsClass = copy.copy(queueTimes)
        i = -1
        for q in queueTimes:
            i += 1
            if q == 0:
                queueTimeAsClass[i] = 0 # first class
            elif q < 5:
                queueTimeAsClass[i] = 1 # second class
            elif q < 30:
                queueTimeAsClass[i] = 2 # third class
            elif q < 60:
                queueTimeAsClass[i] = 3 # fourth class
            elif q < 120:
                queueTimeAsClass[i] = 4 # fifth class
            elif q < 240:
                queueTimeAsClass[i] = 5 # sixth class
            elif q <= 480:
                queueTimeAsClass[i] = 6 # seventh class
            elif q > 480:
                queueTimeAsClass[i] = 7 # eighth class
        return queueTimeAsClass

    queueTimesClass1 = qt_to_class(dataset_forecast['QueueTime'].values)
                
    necessaryFeatures=["Partition","ReqNodes","ReqCPUS","NNodes","Timelimit","sin(Weekday)","cos(Weekday)","sin(Dayminute)","cos(Dayminute)","QueueTime"]

    forecasting_set_categorical = copy.copy(dataset_forecast[necessaryFeatures].values)
    forecasting_set = dataset_forecast[necessaryFeatures].values

    # Compute the size of the forecasting dataset
    forecastsetHeight, forecastsetWidth = np.shape(forecasting_set_categorical)

    #-----------------------------------------------IMPORT OneHotEncoder------------------------------------------------
    from sklearn.preprocessing import OneHotEncoder
    #-----------------------------------------------SKLEARN-------------------------------------------------------------
    onehotencoder_X = OneHotEncoder(categorical_features = [0])
    forecasting_set_label = onehotencoder_X.fit_transform(forecasting_set_categorical).toarray()

    array1=dataset_forecast["QueueTime"].values
    array2 = copy.copy(queueTimesClass1)
    array12=np.append(array1.reshape(array1.size,1),array2.reshape(array1.size,1),axis=1)
    onehotencoder_Y = OneHotEncoder(categorical_features = [1])
    y_forecast_class_label = onehotencoder_Y.fit_transform(array12).toarray()

    # Avoid dummy variable trap
    forecasting_set_ind = np.delete(forecasting_set_label, 0, 1) # delete the first column to avoid dummy v trap

    # -------------------------------------------------------------------------------------------------------------------
    # this number is linked to the number of partitions
    numberIndFeatures = forecasting_set_ind.shape[1]
    # -------------------------------------------------------------------------------------------------------------------

    forecasting_set_new = copy.copy(forecasting_set_ind)
    #forecasting_set_new[:, 0:numberIndFeatures-1] = forecasting_set_ind[:, 0:numberIndFeatures-1]


    #-----------------------------------------------IMPORT MinMaxScaler-------------------------------------------------
    from sklearn.preprocessing import MinMaxScaler
    #-----------------------------------------------SKLEARN-------------------------------------------------------------
    # Feature scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    forecasting_set_scaled = sc.fit_transform(forecasting_set_new[:, 0:numberIndFeatures-1])

    #-----------------------------------------------IMPORT LabelEncoder-------------------------------------------------
    from sklearn.preprocessing import LabelEncoder
    #-----------------------------------------------SKLEARN-------------------------------------------------------------


    #lag = 23 # number of realizations in the past the model looks back in order to predict the future
    forecasting_set_final = copy.copy(forecasting_set_scaled)

    n_classes = 8

    X_forecast = []
    #y_forecast = np.zeros((forecastsetHeight*lag,n_classes))
    for i in range(lag, forecastsetHeight):
        X_forecast.append(forecasting_set_final[i-lag:i, 0:numberIndFeatures-1])
    X_forecast  = np.array(X_forecast)
    y_forecast = np.array(y_forecast_class_label[0:forecastsetHeight-lag,0:n_classes])
    # Reshaping forecasting set
    X_forecast = np.reshape(X_forecast, (X_forecast.shape[0], X_forecast.shape[1], numberIndFeatures-1))


    #laforecast_model="20180927184133.h5"
    latest_model_full_path = models_path +"/"+model_name
    latest_model_full_path_OS = convert_path(latest_model_full_path)

    #-----------------------------------------------IMPORT LOAD_MODEL-------------------------------------------------
    from keras.models import load_model
    #-----------------------------------------------FROM KERAS--------------------------------------------------------

    # ==========================================================================================================
    # load laforecast model
    classifier = load_model(latest_model_full_path_OS)
    # make predictions
    predicted_forecast = classifier.predict(X_forecast)
    # ==========================================================================================================


    #-------------------------------------------------------------------------------------------------------------
    def compute_class_index(pred_softmax):
        output_list_index = []
        for i in range(pred_softmax.shape[0]):
            max_prediction_index = np.argmax(pred_softmax[i, :])
            output_list_index.append(max_prediction_index)
        return output_list_index
    #-------------------------------------------------------------------------------------------------------------

    # Predict forecasting data class
    print('Computing the predicted class...')

    class_forecast = compute_class_index(predicted_forecast)

                                              
    displayFeatures=["Partition","ReqNodes","ReqCPUS","NNodes","Timelimit","Weekday","Dayminute","QueueTime"]
    class_forecast = class_forecast
    Forecast = copy.copy(dataset_forecast_original[displayFeatures])
    indexes_partitions_boole=indexes_partitions==1

    for p in PartitionTypes:
        df_class_forecast = pd.DataFrame(class_forecast, columns=["ForecastLarge"])
        df_class_forecast_partition = df_class_forecast.iloc[indexes_partitions_boole[lag:,3]]
        class_forecast_partition = pd.DataFrame(df_class_forecast_partition.values, columns=[p], index=range(lag+1,Forecast.shape[0]))
        Forecast = Forecast.join(class_forecast_partition)                                          

    f_qt_class = np.zeros((Forecast.shape[0],1))
    for j in range(lag+1,Forecast.shape[0]):
        p = Forecast.iloc[j,0]
        f_qt_class[j] = Forecast[p].iloc[j]
    df_f_qt_class = pd.DataFrame(f_qt_class, columns=["Test QT"])
    Forecast = Forecast.join(df_f_qt_class) 
    displayFeatures=["Partition","ReqNodes","ReqCPUS","NNodes","Timelimit","Weekday","Dayminute","QueueTime","Test QT"]
    Forecast[displayFeatures]


    sz = Forecast.shape[0]
    forecast_full_path = forecast_path+"/"+"f_"+model_name+"_lag="+str(lag)+"_size="+str(sz)+".csv"
    forecast_full_pathOS = convert_path(forecast_full_path)
    Forecast.iloc[lag+1:,:].to_csv(forecast_full_path)


    #-------------------------------------------------------------------------------------------------------------
    # Function that computes the reliability of the model
    def compute_model_reliability(class_index, target):
        indicator = 0
        for z in range(len(class_index)):
            if class_index[z] == int(target[z]):
                indicator += 1
            model_reliability = indicator/len(class_index)
        return model_reliability
    #-------------------------------------------------------------------------------------------------------------

    real_QT_as_class = qt_to_class(Forecast["QueueTime"].values)

    if test_model:

        # Compute reliability of model on forecasting data
        print('Computing the reliability of the forecasted model...')
        #--------------------------------------------OUTPUT-----------------------------------------------------------
        model_reliability_forecast = compute_model_reliability(real_QT_as_class,Forecast["Test QT"].values)
        #--------------------------------------------OUTPUT-----------------------------------------------------------
        print('The reliability of the model on forecasting data is %f'% model_reliability_forecast)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(16,6))
        plt.plot(real_QT_as_class, color="blue", label="real data")
        plt.plot(Forecast["Test QT"].values, color="red", label="forecasting")
        title="TESTING: "+forecasting_set_csv+" | lag="+str(lag)+" "
        plt.title(title)
        plt.ylabel('Queue Time')
        txt = latest_model_full_path_OS + "reliability = %.3f"%model_reliability_forecast
        plt.xlabel(txt)
        plt.legend()
        #plt.show()
        fig.savefig("testing_"+model_name+'.pdf')


    ts = time.time()

    print("Total time needed to forecast the model:")
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Output 1: %s"%latest_model_full_path_OS)
    print("Output 2: %s"%forecast_full_path)
    #print("Output 2: %f"%model_reliability_forecast)


    """
    txt = model_full_path+" ~~ reliability = %.3f"%model_reliability_forecast
    title="FORECASTING: "+forecasting_set_csv+" | lag="+str(lag)+" "
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16,6))
    #ax1 = fig.add_axes((0.1,0.1,0.9,0.9))
    plt.plot(queueTimesClass1[lag-1:-1], label="real data")
    plt.plot(index_forecast, color="red", label="forecasting")
    plt.title(title)
    plt.ylabel('Queue Time')
    plt.xlabel(txt)
    plt.legend()
    plt.show()
    fig.savefig(model_name"_forecast"+'.pdf')
    """

    return forecast_full_path
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^