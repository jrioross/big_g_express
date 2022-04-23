### The functions support multiple notebooks in my Big_G repository

#### Functions I used to prepare the data for my models (not Chris Harrelson's models)

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

#### Functions and processes I'm using for model assessment

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, f1_score, auc, precision_score, recall_score
import plotly.express as px

def cost_scorer(y_test, y_pred, cost_fp = 500, cost_fn = 5000):
    """Use the confusion matrix of a classification model to determine the total costs of false predictions."""
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return (cost_fp*fp) + (cost_fn*fn)

def savings_scorer(y_test, y_pred, save_tp = 5000, cost_fp = -500):
    """
    Use the confusion matrix of a classificaton model to determine the overall savings from implementing the model.
    The assumption is that false negatives are the same expenses the company would have without implementing the model
    and the true negatives are neither profitable nor costly. Therefore, the only influential values for savings are
    true positives (positive saved money) and false positives (negative money, lost from extra expenses).
    """
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return (save_tp*tp) + (cost_fp*fp)

def plotly_roc(y_true, y_pred_probs):
    """
    Create a plotly ROC curve, title with the AUC, and note the optimal threshold.
    Thresholds for each ROC curve point are included in the tooltip.
    """
    fpr, tpr, thresholds = roc_curve(y_true, [p[1] for p in y_pred_probs])
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("The optimum tpr vs. fpr threshold value is:", optimal_threshold)

    fig = px.area(
        x=fpr, 
        y=tpr,
        hover_data=[thresholds],
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', 
                    y='True Positive Rate',
                    hover_data_0 = 'Threshold'),
        width=700, 
        height=500
    )

    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig

def plot_costs_by_threshold(costs_df, multiwindow = False):
    """
    Using a dataframe of thresholds and their total_costs, create a Plotly line chart
    and title with the best threshold and window (if windows are also supplied).
    """
    min_costs = costs_df.nsmallest(1, 'total_costs')['total_costs'].values[0]
    cheapest_threshold = costs_df.nsmallest(1, 'total_costs')['threshold'].values[0]

    if multiwindow:
        cheapest_window = costs_df.nsmallest(1, 'total_costs')['target_window'].values[0]
        fig = px.line(costs_df,
            x = 'threshold',
            y = 'total_costs',
            color='target_window',
            width = 700,
            height = 500,
            title=f'Minimum Total Costs are ${min_costs:,.2f}<br>with a Target Window of {cheapest_window}<br>and a Threshold of {cheapest_threshold}'
        )
    else:
        fig = px.line(costs_df,
            x = 'threshold',
            y = 'total_costs',
            width = 700,
            height = 500,
            title=f'Minimum Total Costs are ${min_costs:,.2f}<br>with a Threshold of {cheapest_threshold}'
        )
    return fig

def plot_savings_by_threshold(savings_df, multiwindow = False):
    """
    Using a dataframe of thresholds and their total_savings, create a Plotly line chart
    and title with the best threshold and window (if windows are also supplied).
    """
    max_savings = savings_df.nlargest(1, 'total_savings')['total_savings'].values[0]
    best_threshold = savings_df.nlargest(1, 'total_savings')['threshold'].values[0]

    if multiwindow:
        best_window = savings_df.nlargest(1, 'total_savings')['target_window'].values[0]
        fig = px.line(savings_df,
            x = 'threshold',
            y = 'total_savings',
            color='target_window',
            width = 700,
            height = 500,
            title=f'Maximum Total Savings are ${max_savings:,.2f}<br>with a Target Window of {best_window}<br>and a Threshold of {best_threshold}'
        )
    else:
        fig = px.line(savings_df,
            x = 'threshold',
            y = 'total_savings',
            width = 700,
            height = 500,
            title=f'Maximum Total Savings are ${max_savings:,.2f}<br>with a Threshold of {best_threshold}'
        )
    return fig

def quick_report(y_true, y_pred_probs, threshold, cost_fp=500, cost_fn=5000, save_tp = 5000):
    """Get a quick model report given true values, prediction probabilities, and a threshold."""
    save_fp = -1*cost_fp
    preds = [p[1] > threshold for p in y_pred_probs]
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    print("confusion matrix:\n", confusion_matrix(y_true, preds))
    print("False Positives: ", fp, "\nFalse Negatives: ", fn)
    print("f1 score: ", f1_score(y_true, preds))
    print("precision score: ", precision_score(y_true, preds))
    print("recall score: ", recall_score(y_true, preds))
    print("Total Costs: ",  "${:,.2f}".format(cost_scorer(y_true, preds, cost_fp=cost_fp, cost_fn=cost_fn)))
    print("Total Savings: ",  "${:,.2f}".format(savings_scorer(y_true, preds, save_tp=save_tp, cost_fp=save_fp)))
    return

#### Adapting Michael Holloway's code for confusion matrices

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
        
        plt.annotate(text ='Savings: ${:,.2f}'.format(5000*cm[1][1]-500*cm[0][1]), 
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

