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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix

def confusion_matrix_demo(labels = ['negative','positive'], metric = None, figsize = (6,6)):
    fontsize = 14

    fig, ax = plt.subplots(figsize = figsize)
    plt.vlines(x = [0,1,2], ymin = 0, ymax = 2)
    plt.hlines(y = [0,1,2], xmin = 0, xmax = 2)

    plt.annotate(text =labels[0], xy = (0.5, 2.05), va = 'bottom', ha = 'center', fontsize = fontsize)
    plt.annotate(text =labels[1], xy = (1.5, 2.05), va = 'bottom', ha = 'center', fontsize = fontsize)
    plt.annotate(text =labels[0], xy = (-0.05, 1.5), va = 'center', ha = 'right', fontsize = fontsize, rotation = 90)
    plt.annotate(text =labels[1], xy = (-0.05, 0.5), va = 'center', ha = 'right', fontsize = fontsize, rotation = 90)

    plt.annotate(text ='Predicted', xy = (1, 2.25), va = 'bottom', ha = 'center', fontsize = fontsize + 2, fontweight = 'bold')
    plt.annotate(text ='Actual', xy = (-0.25, 1), va = 'center', ha = 'right', fontsize = fontsize + 2, fontweight = 'bold', rotation = 90)

    plt.annotate(text = 'True Negatives\n (TN)', xy = (0.5, 1.5), fontsize = fontsize, ha = 'center', va = 'center')
    plt.annotate(text = 'False Negatives\n (FN)', xy = (0.5, 0.5), fontsize = fontsize, ha = 'center', va = 'center')   
    plt.annotate(text = 'False Positives\n (FP)', xy = (1.5, 1.5), fontsize = fontsize, ha = 'center', va = 'center')   
    plt.annotate(text = 'True Positives\n (TP)', xy = (1.5, 0.5), fontsize = fontsize, ha = 'center', va = 'center')    
    
    plt.ylim(-0.2, 2.5)
    plt.xlim(-0.5, 2.1)
    
    plt.axis('off') 
    

def plot_confusion_matrix(y_true, y_pred, labels = [0,1], metric = None, figsize = (6,6)):
    fontsize = 14

    fig, ax = plt.subplots(figsize = figsize)
    plt.vlines(x = [0,1,2], ymin = 0, ymax = 2)
    plt.hlines(y = [0,1,2], xmin = 0, xmax = 2)

    plt.annotate(text =labels[0], xy = (0.5, 2.05), va = 'bottom', ha = 'center', fontsize = fontsize)
    plt.annotate(text =labels[1], xy = (1.5, 2.05), va = 'bottom', ha = 'center', fontsize = fontsize)
    plt.annotate(text =labels[0], xy = (-0.05, 1.5), va = 'center', ha = 'right', fontsize = fontsize, rotation = 90)
    plt.annotate(text =labels[1], xy = (-0.05, 0.5), va = 'center', ha = 'right', fontsize = fontsize, rotation = 90)

    plt.annotate(text ='Predicted', xy = (1, 2.25), va = 'bottom', ha = 'center', fontsize = fontsize + 2, fontweight = 'bold')
    plt.annotate(text ='Actual', xy = (-0.25, 1), va = 'center', ha = 'right', fontsize = fontsize + 2, fontweight = 'bold', rotation = 90)

    cm = confusion_matrix(y_true, y_pred)

    for i in range(2):
        for j in range(2):
            plt.annotate(text =cm[j][i], xy = (0.5 + i, 1.5 - j), fontsize = fontsize + 4, ha = 'center', va = 'center')


    plt.ylim(-0.2, 2.5)
    plt.xlim(-0.5, 2.1)
    
    if metric == 'accuracy':
        ax.add_patch(Rectangle(xy = (0,1), width = 1, height = 1, color = 'lightgreen'))
        ax.add_patch(Rectangle(xy = (1,0), width = 1, height = 1, color = 'lightgreen'))
        ax.add_patch(Rectangle(xy = (0,0), width = 1, height = 1, color = 'coral'))
        ax.add_patch(Rectangle(xy = (1,1), width = 1, height = 1, color = 'coral'))
        
        plt.annotate(text ='Accuracy: {}'.format(round((cm[0][0] + cm[1][1]) / cm.sum(),3)), 
                     xy = (1, -0.1), ha = 'center', va = 'top', fontsize = fontsize + 2)
        
    if metric == 'sensitivity':
        ax.add_patch(Rectangle(xy = (1,0), width = 1, height = 1, color = 'lightgreen'))
        ax.add_patch(Rectangle(xy = (0,0), width = 1, height = 1, color = 'coral'))
        
        plt.annotate(text ='Sensitivity: {}'.format(round((cm[1][1]) / cm[1].sum(),3)), 
                     xy = (1, -0.1), ha = 'center', va = 'top', fontsize = fontsize + 2)
                     
    if metric == 'recall':
        ax.add_patch(Rectangle(xy = (1,0), width = 1, height = 1, color = 'lightgreen'))
        ax.add_patch(Rectangle(xy = (0,0), width = 1, height = 1, color = 'coral'))
        
        plt.annotate(text ='Recall: {}'.format(round((cm[1][1]) / cm[1].sum(),3)), 
                     xy = (1, -0.1), ha = 'center', va = 'top', fontsize = fontsize + 2)
        
    if metric == 'specificity':
        ax.add_patch(Rectangle(xy = (0,1), width = 1, height = 1, color = 'lightgreen'))
        ax.add_patch(Rectangle(xy = (1,1), width = 1, height = 1, color = 'coral'))
        
        plt.annotate(text ='Specificity: {}'.format(round((cm[0][0]) / cm[0].sum(),3)), 
                     xy = (1, -0.1), ha = 'center', va = 'top', fontsize = fontsize + 2)
        
    if metric == 'f1':
        ax.add_patch(Rectangle(xy = (1,0), width = 1, height = 1, color = 'lightgreen'))
        ax.add_patch(Rectangle(xy = (1,1), width = 1, height = 1, color = 'coral'))
        ax.add_patch(Rectangle(xy = (0,0), width = 1, height = 1, color = 'coral'))
        
        plt.annotate(text ='F1: {}'.format(round(2*(cm[1][1]) / (2*cm[1][1] + cm[0,1] + cm[1,0]),3)), 
                     xy = (1, -0.1), ha = 'center', va = 'top', fontsize = fontsize + 2)   
                     
    if metric == 'precision':
        ax.add_patch(Rectangle(xy = (1,0), width = 1, height = 1, color = 'lightgreen'))
        ax.add_patch(Rectangle(xy = (1,1), width = 1, height = 1, color = 'coral'))
        
        plt.annotate(text ='Precision: {}'.format(round((cm[1][1]) / cm[:,1].sum(),3)), 
                     xy = (1, -0.1), ha = 'center', va = 'top', fontsize = fontsize + 2)   

    if metric == 'savings':
        ax.add_patch(Rectangle(xy = (0,1), width = 1, height = 1, color = 'lightgrey'))
        ax.add_patch(Rectangle(xy = (1,0), width = 1, height = 1, color = 'lightgreen'))
        ax.add_patch(Rectangle(xy = (0,0), width = 1, height = 1, color = 'lightgrey'))
        ax.add_patch(Rectangle(xy = (1,1), width = 1, height = 1, color = 'coral'))
        
        plt.annotate(text ='Savings: ${:,.2f}'.format(5000*cm[0][1]-500*cm[1][1]), 
                     xy = (1, -0.1), ha = 'center', va = 'top', fontsize = fontsize + 2)

    plt.axis('off')
    
    
def plot_confusion_matrix_multiclass(y_true, y_pred, labels, figsize = (6,6)):
    fontsize = 14

    fig, ax = plt.subplots(figsize = figsize)
    plt.vlines(x = range(len(labels) + 2), ymin = 0, ymax = len(labels))
    plt.hlines(y = range(len(labels) + 2), xmin = 0, xmax = len(labels))

    for i, label in enumerate(labels):
        plt.annotate(text = label, xy = (i + 0.5, len(labels) + 0.05), va = 'bottom', ha = 'center', fontsize = fontsize)
        plt.annotate(text = label, xy = (-0.05, len(labels) - i - 0.5), va = 'center', ha = 'right', fontsize = fontsize, rotation = 90)

        
        
    
    #plt.annotate(text =labels[0], xy = (0.5, 2.05), va = 'bottom', ha = 'center', fontsize = fontsize)
    #plt.annotate(text =labels[1], xy = (1.5, 2.05), va = 'bottom', ha = 'center', fontsize = fontsize)
    #plt.annotate(text =labels[0], xy = (-0.05, 1.5), va = 'center', ha = 'right', fontsize = fontsize, rotation = 90)
    #plt.annotate(text =labels[1], xy = (-0.05, 0.5), va = 'center', ha = 'right', fontsize = fontsize, rotation = 90)

    plt.annotate(text ='Predicted', xy = (len(labels) / 2, len(labels) + 0.25), va = 'bottom', ha = 'center', fontsize = fontsize + 2, fontweight = 'bold')
    plt.annotate(text ='Actual', xy = (-0.25, len(labels) / 2), va = 'center', ha = 'right', fontsize = fontsize + 2, fontweight = 'bold', rotation = 90)

    cm = confusion_matrix(y_true, y_pred)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.annotate(text =cm[j][i], xy = (0.5 + i, len(labels) - 0.5 - j), fontsize = fontsize + 4, ha = 'center', va = 'center')


    plt.ylim(-0.2, len(labels) + 0.5)
    plt.xlim(-0.5, len(labels) + 0.1)
    
    plt.axis('off')

