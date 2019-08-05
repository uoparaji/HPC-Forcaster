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



def parse_raw_data(rawdata_name_dot_csv):
    #-----------------------------------------------IMPORT TIME--------------------------------------------------------
    import time
    #-----------------------------------------------IMPORT TIME--------------------------------------------------------
    start_time = time.time()

    data_and_scripts_location = all_paths()
    raw_data_path = data_and_scripts_location["Data/raw"]

    #-----------------------------------------------IMPORT PANDAS--------------------------------------------------------
    import pandas as pd
    #-----------------------------------------------IMPORT PANDAS--------------------------------------------------------
    DF = pd.read_csv(raw_data_path + "/" + rawdata_name_dot_csv , sep='|', low_memory = False)

    print("Start parsing the data...")

    #print(DF.columns.values)
    featureNames = DF.columns.values
    numFeatures = len(DF.columns.values)

    requiredFeatures = ['Partition','ReqNodes','ReqCPUS','NNodes','Timelimit','Submit','Start','End','Eligible','QueueTime']

    out = set(featureNames).intersection(requiredFeatures)
    if (len(out)!=len(requiredFeatures)):
        missing = [feature for feature in requiredFeatures if feature not in featureNames]
        raise ValueError("The following Features: %s are missing (or mispelled) in the provided dataset" % missing)

    workingDataset = DF.loc[:,requiredFeatures]

    # Convert q times to integers (in seconds)
    print("------------------------------------Parse the feature 'QueueTime'-------------------------------------")
    QTseries = workingDataset.loc[:,"QueueTime"]
    # A better strategy to deal with non formatted time is needed to make the algorithm more robust
    ld = QTseries.str.contains("day")
    if any(ld): 
        print("special convertion from days is needed \ndetecting number of days...")
        QTdays = QTseries.loc[ld]
        days = []
        for day in QTdays:
            dayFormat = day.split(" ")
            days.append(int(dayFormat[0])*24*60) #convert days into minutes
            #days.append("%i:00:00"%dayValue)
        print("No. %i entries detected with unkown time format of type 'days' for qTimes."%ld.values.sum())
        QTseries.loc[ld] = days

    #convert to minutes and neglect the seconds to keep integer format
    QTseries.loc[~ld] = QTseries.loc[~ld].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    print("-------------------------------------done-------------------------------------------------------------")


    # Convert the time limits to integers (in seconds)
    print("------------------------------------Parse the feature 'Timelimit'-------------------------------------")
    TLseries = workingDataset.loc[:,"Timelimit"]
    ltl_u = TLseries.str.contains("UNLIMITED")
    ltl_p = TLseries.str.contains("Partition_Limit")
    ltl_d = TLseries.str.contains("-")
    ltd_up = (ltl_u | ltl_p)
    ltl_udp = (ltl_u | ltl_d | ltl_p)

    if any(ltl_p): 
        print("No. %i entries of 'Partition_Limit' type detected. Partition_Limit will be converted into a large integer (~3 years)"%ltl_p.values.sum())
        TLseries.loc[ltl_p] = int(10**8)


    if any(ltl_u): 
        print("No. %i entries of 'UNLIMITED' type detected. 'UNLIMITED' time will be converted into a large integer (~3 years)"%ltl_u.values.sum())
        TLseries.loc[ltl_u] = int(10**8)


    if any(ltl_d):
        print("No. %i entries expressed in days. These values will be converted in minutes."%ltl_d.values.sum())
        TLseries.loc[ltl_d] = TLseries.loc[ltl_d].str.replace('-',':')


    #-----------------------------------------------IMPORT NUMPY--------------------------------------------------------
    import numpy as np
    #-----------------------------------------------IMPORT NUMPY--------------------------------------------------------
    y=np.array(range(0,int(ltl_udp.size)))
    Zs = pd.Series(["00:"]*y[~ltl_udp.values].size, index=y[~ltl_udp.values])
    TLseries.loc[~ltl_udp] = Zs.str.cat(TLseries.loc[~ltl_udp])

    #convert to minutes and neglect the seconds
    TLseries.loc[~ltd_up] = TLseries.loc[~ltd_up].str.split(':').apply(lambda x: int(x[0]) * 24 * 60 + int(x[1]) * 60 + int(x[2]))
    print("-------------------------------------done-------------------------------------------------------------")

    # Convert the submission time into two features (in seconds)
    print("------------------------------------Parse 'Submission time'-------------------------------------")
    STseries = workingDataset.loc[:,"Submit"]

    STseries=STseries.str.replace('T',' ')
    STseries=pd.to_datetime(STseries) # this turns the data into pandas Timestamps
    print("...Converting submit date to weekday")
    workingDataset["Weekday"]=STseries.apply(lambda x: x.weekday())
    print("...Converting submit date to minutes within the day")
    workingDataset["Dayminute"]=STseries.apply(lambda x: x.hour*60+x.minute)

    # compute week number
    week_array_base=workingDataset["Weekday"].values
    week_array = np.ones(len(week_array_base)+1)*week_array_base[0]
    week_array[1:] = workingDataset["Weekday"].values
    new_array_diff=np.diff(week_array)
    l_nv = new_array_diff<0
    d_nv = 1*l_nv
    weeknumber = d_nv.cumsum()+1
    workingDataset["WeekNumber"]=pd.Series(weeknumber)
    totalweeks = workingDataset["WeekNumber"].values[-1]

    days_in_week=7
    workingDataset["sin(Weekday)"]=STseries.apply(lambda x: np.sin(2*np.pi*x.weekday()/days_in_week))
    workingDataset["cos(Weekday)"]=STseries.apply(lambda x: np.cos(2*np.pi*x.weekday()/days_in_week))  

    minutes_in_day = 24*60
    workingDataset["sin(Dayminute)"]=STseries.apply(lambda x: np.sin(2*np.pi*(x.hour*60+x.minute)/minutes_in_day))
    workingDataset["cos(Dayminute)"]=STseries.apply(lambda x: np.cos(2*np.pi*(x.hour*60+x.minute)/minutes_in_day))  

    cols = ['Partition','ReqNodes','ReqCPUS','NNodes','Timelimit','Submit',"Weekday","WeekNumber","sin(Weekday)","cos(Weekday)","Dayminute","sin(Dayminute)","cos(Dayminute)",'QueueTime']

    print("...Compute size of the dataset and number of weeks")
    sheetLength = len(workingDataset["WeekNumber"])
    parsed_data_path = data_and_scripts_location["Data/parsed"]
    
    #--------------------------------------------OUTPUT------------------------------------------------------------------
    output = parsed_data_path + "/parsed_data_size="+str(sheetLength)+"_weeks="+str(totalweeks)+".csv"
    #--------------------------------------------OUTPUT------------------------------------------------------------------
    
    workingDataset[cols].to_csv(output, sep=",", index = False)

    print("-------------------------------------done-------------------------------------------------------------")

    a,b=workingDataset.shape
    print("Total time needed to parse %i entries:" % a)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print("Output: %s"%output)
    
    return output




def split_original_parsed_dataset(number_weeks_training, name_parsed_dataset_csv):
    #-----------------------------------------------IMPORT TIME--------------------------------------------------------
    import time
    #-----------------------------------------------IMPORT TIME--------------------------------------------------------
    start_time = time.time()

    data_and_scripts_location = all_paths()
    parsed_data_path = data_and_scripts_location["Data/parsed"]

    #-----------------------------------------------IMPORT PANDAS--------------------------------------------------------
    import pandas as pd
    #-----------------------------------------------IMPORT PANDAS--------------------------------------------------------

    DF = pd.read_csv(parsed_data_path + "/" + name_parsed_dataset_csv, sep=',', low_memory = False)

    print("Start splitting the dataset...")

    requiredFeatures = ['Partition','ReqNodes','ReqCPUS','NNodes','Timelimit','Submit',"Weekday","WeekNumber","sin(Weekday)","cos(Weekday)","Dayminute","sin(Dayminute)","cos(Dayminute)",'QueueTime']
    workingDataset = DF.loc[:,requiredFeatures]
    necessaryFeatures = ['Partition','ReqNodes','ReqCPUS','NNodes','Timelimit',"Weekday","sin(Weekday)","cos(Weekday)","Dayminute","sin(Dayminute)","cos(Dayminute)",'QueueTime']
    totalweeks = workingDataset["WeekNumber"].values[-1]
    totalsize = len(workingDataset["WeekNumber"])
    

    # Make training dataset
    # weeks_training = 5
    weeks_training = number_weeks_training
    l_wn = workingDataset["WeekNumber"].values <= weeks_training
    sheetSize = len(workingDataset["WeekNumber"].loc[l_wn])
    #--------------------------------------------OUTPUT------------------------------------------------------------------
    output = parsed_data_path + "/training_set_size="+ str(sheetSize) +"_weeks="+ str(weeks_training) +".csv"
    #--------------------------------------------OUTPUT------------------------------------------------------------------
    workingDataset[necessaryFeatures].loc[l_wn].to_csv(output, sep=",",index=False)

    print("\nDateset consisting of %i entries and %i weeks split into:\n 1 training set of %i entries equivalent to %i weeks of job submissions\n"%(totalsize,totalweeks,sheetSize,weeks_training))

    # Make test sets
    number_test_sets = len(range(weeks_training+1,int(totalweeks)))
    print(" %i Test sets for each of the following weeks:"% number_test_sets)
    for w in range(weeks_training+1,int(totalweeks)):
        l_wn_tests = workingDataset["WeekNumber"].values == w
        sheetSize = len(workingDataset["WeekNumber"].loc[l_wn_tests])
        workingDataset[necessaryFeatures].loc[l_wn_tests].to_csv(parsed_data_path + "/test_set_size="+ str(sheetSize) +"_week="+ str(w) +".csv", sep=",",index=False)
        print(" %i entries equivalent to %i week of job submissions"%(sheetSize,1))


    print("Splitting complete. Total time needed:")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    print("Ouput: %s"%output)
    
    return output