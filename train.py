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
    }
    return data_and_scripts_location


def convert_path(full_path):
    #make sure the path is OS indipendent
    from os.path import expanduser
    from os.path import join
    home = expanduser("~")
    full_path_w=full_path.split("/")
    conv_full_path = home
    for f in full_path_w[1:]:
        conv_full_path = join(conv_full_path,f)
    return conv_full_path


def train_lstm(training_set_csv, lag=50, optim='sgd', epo=37, layer1 = 227, layer2 = 97):
    #training_set_csv = "training_set_size=206908_weeks=100.csv"
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

    dataset_train = pd.read_csv(convert_path(data_path + "/" + training_set_csv))
    dataset_train['Partition'] = dataset_train['Partition'].astype('category')
    dataset_train['Partition'] = dataset_train['Partition'].cat.codes

    #-----------------------------------------------IMPORT COPY--------------------------------------------------------
    import copy
    #-----------------------------------------------IMPORT COPY--------------------------------------------------------

    queueTimes = dataset_train['QueueTime'].values
    queueTimesClass1 = copy.copy(queueTimes)
    i = -1
    for q in queueTimes:
        i += 1
        if q == 0:
            queueTimesClass1[i] = 0 # first class
        elif q < 5:
            queueTimesClass1[i] = 1 # second class
        elif q < 30:
            queueTimesClass1[i] = 2 # third class
        elif q < 60:
            queueTimesClass1[i] = 3 # fourth class
        elif q < 120:
            queueTimesClass1[i] = 4 # fifth class
        elif q < 240:
            queueTimesClass1[i] = 5 # sixth class
        elif q < 480:
            queueTimesClass1[i] = 6 # seventh class
        elif q > 480:
            queueTimesClass1[i] = 7 # eighth class
            
    necessaryFeatures=["Partition","ReqNodes","ReqCPUS","NNodes","Timelimit","sin(Weekday)","cos(Weekday)","sin(Dayminute)","cos(Dayminute)","QueueTime"]

    training_set_categorical = copy.copy(dataset_train[necessaryFeatures].values)
    training_set = dataset_train[necessaryFeatures].values

    #-----------------------------------------------IMPORT NUMPY--------------------------------------------------------
    import numpy as np
    #-----------------------------------------------IMPORT NUMPY--------------------------------------------------------
    # Compute the size of the training dataset
    trainsetHeight, trainsetWidth = np.shape(training_set_categorical)

    #-----------------------------------------------IMPORT OneHotEncoder------------------------------------------------
    from sklearn.preprocessing import OneHotEncoder
    #-----------------------------------------------SKLEARN-------------------------------------------------------------
    onehotencoder_X = OneHotEncoder(categorical_features = [0])
    training_set_label = onehotencoder_X.fit_transform(training_set_categorical).toarray()

    array1=dataset_train["QueueTime"].values
    array2 = copy.copy(queueTimesClass1)
    array12=np.append(array1.reshape(array1.size,1),array2.reshape(array1.size,1),axis=1)
    onehotencoder_Y = OneHotEncoder(categorical_features = [1])
    y_train_class_label = onehotencoder_Y.fit_transform(array12).toarray()

    # Avoid dummy variable trap
    training_set_ind = np.delete(training_set_label, 0, 1) # delete the first column to avoid dummy v trap

    # -------------------------------------------------------------------------------------------------------------------
    # this number is linked to the number of partitions
    numberIndFeatures = training_set_ind.shape[1]
    # -------------------------------------------------------------------------------------------------------------------

    training_set_new = copy.copy(training_set_ind)
    #training_set_new[:, 0:numberIndFeatures-1] = training_set_ind[:, 0:numberIndFeatures-1]


    #-----------------------------------------------IMPORT MinMaxScaler-------------------------------------------------
    from sklearn.preprocessing import MinMaxScaler
    #-----------------------------------------------SKLEARN-------------------------------------------------------------
    # Feature scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set_new[:, 0:numberIndFeatures-1])

    #-----------------------------------------------IMPORT LabelEncoder-------------------------------------------------
    from sklearn.preprocessing import LabelEncoder
    #-----------------------------------------------SKLEARN-------------------------------------------------------------

    #training_set_new_class = training_set_new[:, numberIndFeatures-1]
    #training_set_new_class_reshape = training_set_new_class.reshape(training_set_new_class.size, 1)
    #training_set_scaled = np.append(training_set_scaled, training_set_new_class_reshape, axis = 1)
    #labelencoder = LabelEncoder()
    #training_set_scaled[:, numberIndFeatures-1] = labelencoder.fit_transform(training_set_scaled[:, numberIndFeatures-1])

    #training_size = int(np.ceil(training_set_scaled[:,0].size*1.0))

    #lag = 23 # number of realizations in the past the model looks back in order to predict the future
    #training_set_final = training_set_scaled[0:training_size+1,:]
    training_set_final = copy.copy(training_set_scaled)

    n_classes = 8

    X_train = []
    #y_train = np.zeros((trainsetHeight*lag,n_classes))
    for i in range(lag, trainsetHeight):
        X_train.append(training_set_final[i-lag:i, 0:numberIndFeatures-1])
    X_train  = np.array(X_train)
    y_train = np.array(y_train_class_label[0:trainsetHeight-lag,0:n_classes])
    # Reshaping training set
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], numberIndFeatures-1))

    #-----------------------------------------------IMPORT KERAS-------------------------------------------------
    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras import optimizers
    #-----------------------------------------------IMPORT KERAS-------------------------------------------------

    classifier = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    classifier.add(LSTM(units = layer1, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
    # Adding some dropout to help reduce overfitting the model
    classifier.add(Dropout(0.2))
    # Adding the second LSTM layer and some Dropout regularisation
    classifier.add(LSTM(units = layer2))
    # Adding some dropout to help reduce overfitting the model
    classifier.add(Dropout(0.2))
    # Adding the output layer with a softmax activation function for multi classification
    classifier.add(Dense(n_classes, activation='softmax'))
    # Compiling the RNN
    #classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #my_sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #my_rmsdrop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    classifier.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    # Fitting model to data
    classifier.fit(X_train, y_train, epochs = epo, batch_size = 32, shuffle=True)
    # Predicting training data
    predicted_train = classifier.predict(X_train)

    #-------------------------------------------------------------------------------------------------------------
    def compute_class_index(pred_softmax):
        output_list_index = []
        for i in range(pred_softmax.shape[0]):
            max_prediction_index = np.argmax(pred_softmax[i, :])
            output_list_index.append(max_prediction_index)
        return output_list_index
    #-------------------------------------------------------------------------------------------------------------

    # Predict training data class
    print('Computing the predicted class...')

    index_train = compute_class_index(predicted_train)

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

    # Compute reliability of model on training data
    print('Computing the reliability of the trained model...')
    #--------------------------------------------OUTPUT-----------------------------------------------------------
    model_reliability_train = compute_model_reliability(index_train, queueTimesClass1)
    #--------------------------------------------OUTPUT-----------------------------------------------------------
    print('The reliability of the model on training data is %f'% model_reliability_train)

    ts = time.time()
    model_file_name = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S') + '.h5'
    model_full_path = models_path + "/" + model_file_name
    #--------------------------------------------OUTPUT-----------------------------------------------------------
    model_full_path_OS=convert_path(model_full_path)
    #--------------------------------------------OUTPUT-----------------------------------------------------------

    classifier.save(model_full_path_OS)  # creates a HDF5 file     

    print("Total time needed to train the model:")
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Output 1: %s"%model_full_path)
    print("Output 2: %f"%model_reliability_train)



    txt = model_full_path+" ~~ reliability = %.3f"%model_reliability_train
    title=training_set_csv+" | lag="+str(lag)+" | optim="+optim+" | epochs="+str(epo)+" | l1="+str(layer1)+" | l2="+str(layer2) +" | rel=%.3f"%model_reliability_train
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16,6))
    #ax1 = fig.add_axes((0.1,0.1,0.9,0.9))
    plt.plot(queueTimesClass1, label="real data")
    plt.plot(index_train, color="red", label="training")
    plt.title(title)
    plt.ylabel('Queue Time')
    plt.xlabel(txt)
    plt.legend()
    plt.show()
    fig.savefig(model_file_name+'.pdf')

    return fig, model_reliability_train, model_full_path_OS

