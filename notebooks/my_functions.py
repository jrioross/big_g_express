import numpy as np
import pandas as pd
import datetime as dt

def create_target_window(df, both_derate_types = False, target_window_hours = 48):

    df2 = df.copy()
    df2 = df2.sort_values(['EquipmentID', 'EventTimeStamp']).reset_index(drop = True)

    if both_derate_types:

        increment_check_either = (
    
                (df2['spn'].shift() == 5246)  # full derate in above row
                | 
                ((df2['spn'].shift() == 1569) & (df2['fmi'].shift() == 31))   # partial derate in above row
                | 
                (df2['EquipmentID'] != df2['EquipmentID'].shift())    # current row is different truck from previous row

            )

        df2['eventGroup'] = increment_check_either.cumsum()
        eventGroupMaxIndexTransform = df2.groupby('eventGroup')['EventTimeStamp'].transform('idxmax') # get the max index of each event group
        eventGroupEndDerate = (     
                (
                    (df2.loc[eventGroupMaxIndexTransform, 'spn'] == 5246) # full derate
                    |
                    ((df2.loc[eventGroupMaxIndexTransform, 'spn'] == 1569) & (df2.loc[eventGroupMaxIndexTransform, 'fmi'] == 31)) # partial derate
                )
                .reset_index(drop = True)
            )       # check whether or not each event group ends with a derate

    else:    

        increment_check_full_only = (

                (df2['spn'].shift() == 5246)  # full derate in above row
                | 
                (df2['EquipmentID'] != df2['EquipmentID'].shift())    # current row is different truck from previous row

            )

        df2['eventGroup'] = increment_check_full_only.cumsum()
        eventGroupMaxIndexTransform = df2.groupby('eventGroup')['EventTimeStamp'].transform('idxmax')   # get the max index of each event group
        eventGroupEndDerate = (df2.loc[eventGroupMaxIndexTransform, 'spn'] == 5246).reset_index(drop = True) # check whether or not each event group ends with a derate

    df2['timeTillLast'] = df2.groupby('eventGroup')['EventTimeStamp'].transform(max) - df2['EventTimeStamp']
    df2['target'] = (df2['timeTillLast'] < dt.timedelta(hours = target_window_hours)) & eventGroupEndDerate
    
    return df2

def stratifier(df, trucks = False, breaks = [0.6, 0.8, 1.0]):
    df2 = df.copy()
    if trucks:
        conditions_list = [df2['EventTimeStamp'].le(df.groupby('EquipmentID')['EventTimeStamp'].transform(lambda x: x.quantile(val))) for val in breaks]
        choice_list = ['train', 'test', 'validate']

        df2['train_test_val'] = np.select(conditions_list, choice_list)
    else:
        df2['train_test_val'] = pd.qcut(df2['EventTimeStamp'], 
                                          q = [0] + breaks, 
                                          labels = ['train', 'test', 'validation']).astype(str)
    return df2