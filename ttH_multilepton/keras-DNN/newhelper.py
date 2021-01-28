import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import optparse, json, argparse, math
import ROOT
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import time
def newoverfitting(model, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights):
        print(Y_train.shape)
        print(result_probs.shape)
        wt_train_ttH_sample =[]
        wt_train_Other_sample =[]
        wt_train_ttW_sample =[]
        wt_train_tHQ_sample =[]
        wt_train_newphysics_sample =[]
         #Arrays to store all ttH values
        y_scores_train_ttH_sample_ttHnode = []
        y_scores_train_Other_sample_ttHnode = []
        y_scores_train_ttW_sample_ttHnode = []
        y_scores_train_tHQ_sample_ttHnode = []
        y_scores_train_newphysics_sample_ttHnode = []
        # Arrays to store ttH categorised event values
        y_scores_train_ttH_sample_ttH_categorised = []
        y_scores_train_Other_sample_ttH_categorised = []
        y_scores_train_ttW_sample_ttH_categorised = []
        y_scores_train_tHQ_sample_ttH_categorised = []
        y_scores_train_newphysics_sample_ttH_categorised = []
        # Arrays to store all Other node values
        y_scores_train_ttH_sample_Othernode = []
        y_scores_train_Other_sample_Othernode = []
        y_scores_train_ttW_sample_Othernode = []
        y_scores_train_tHQ_sample_Othernode = []
        y_scores_train_newphysics_sample_Othernode = []
        # Arrays to store Other categorised event values
        y_scores_train_ttH_sample_Other_categorised = []
        y_scores_train_Other_sample_Other_categorised = []
        y_scores_train_ttW_sample_Other_categorised = []
        y_scores_train_tHQ_sample_Other_categorised = []
        y_scores_train_newphysics_sample_Other_categorised = []
        # Arrays to store all ttW node values
        y_scores_train_ttH_sample_ttWnode = []
        y_scores_train_Other_sample_ttWnode = []
        y_scores_train_ttW_sample_ttWnode = []
        y_scores_train_tHQ_sample_ttWnode = []
        y_scores_train_newphysics_sample_ttWnode = []
        # Arrays to store ttW categorised events
        y_scores_train_ttH_sample_ttW_categorised = []
        y_scores_train_Other_sample_ttW_categorised = []
        y_scores_train_ttW_sample_ttW_categorised = []
        y_scores_train_tHQ_sample_ttW_categorised = []
        y_scores_train_newphysics_sample_ttW_categorised = []
        # Arrays to store all tHQ node values
        y_scores_train_ttH_sample_tHQnode = []
        y_scores_train_Other_sample_tHQnode = []
        y_scores_train_ttW_sample_tHQnode = []
        y_scores_train_tHQ_sample_tHQnode = []
        y_scores_train_newphysics_sample_tHQnode = []
        # Arrays to store tHQ categorised events
        y_scores_train_ttH_sample_tHQ_categorised = []
        y_scores_train_Other_sample_tHQ_categorised = []
        y_scores_train_ttW_sample_tHQ_categorised = []
        y_scores_train_tHQ_sample_tHQ_categorised = []
        y_scores_train_newphysics_sample_tHQ_categorised = []

        # Arrays to store all newphysics node values
        y_scores_train_ttH_sample_newphysicsnode = []
        y_scores_train_Other_sample_newphysicsnode = []
        y_scores_train_ttW_sample_newphysicsnode = []
        y_scores_train_tHQ_sample_newphysicsnode = []
        y_scores_train_newphysics_sample_newphysicsnode = []
        # Arrays to store newphysics categorised events
        y_scores_train_ttH_sample_newphysics_categorised = []
        y_scores_train_Other_sample_newphysics_categorised = []
        y_scores_train_ttW_sample_newphysics_categorised = []
        y_scores_train_tHQ_sample_newphysics_categorised = []
        y_scores_train_newphysics_sample_newphysics_categorised = []




        for i in range(len(result_probs)):
            #print(result_probs[i][0])
            train_event_weight = train_weights[i]
            if Y_train[i][0] == 1:
                wt_train_ttH_sample.append(train_event_weight)
                y_scores_train_ttH_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_ttH_sample_Othernode.append(result_probs[i][1])
                y_scores_train_ttH_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_ttH_sample_tHQnode.append(result_probs[i][3])
                y_scores_train_ttH_sample_newphysicsnode.append(result_probs[i][4])
                # Get index of maximum argument.
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_ttH_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_ttH_sample_Other_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_ttH_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_ttH_sample_tHQ_categorised.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 4:
                    y_scores_train_ttH_sample_newphysics_categorised.append(result_probs[i][4])

            if Y_train[i][1] == 1:
                wt_train_Other_sample.append(train_event_weight)
                y_scores_train_Other_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_Other_sample_Othernode.append(result_probs[i][1])
                y_scores_train_Other_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_Other_sample_tHQnode.append(result_probs[i][3])
                y_scores_train_Other_sample_newphysicsnode.append(result_probs[i][4])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_Other_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_Other_sample_Other_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_Other_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_Other_sample_tHQ_categorised.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 4:
                    y_scores_train_Other_sample_newphysics_categorised.append(result_probs[i][4])

            if Y_train[i][2] == 1:
                wt_train_ttW_sample.append(train_event_weight)
                y_scores_train_ttW_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_ttW_sample_Othernode.append(result_probs[i][1])
                y_scores_train_ttW_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_ttW_sample_tHQnode.append(result_probs[i][3])
                y_scores_train_ttW_sample_newphysicsnode.append(result_probs[i][4])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_ttW_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_ttW_sample_Other_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_ttW_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_ttW_sample_tHQ_categorised.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 4:
                    y_scores_train_ttW_sample_newphysics_categorised.append(result_probs[i][4])
                    
            if Y_train[i][3] == 1:
                wt_train_tHQ_sample.append(train_event_weight)
                y_scores_train_tHQ_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_tHQ_sample_Othernode.append(result_probs[i][1])
                y_scores_train_tHQ_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_tHQ_sample_tHQnode.append(result_probs[i][3])
                y_scores_train_tHQ_sample_newphysicsnode.append(result_probs[i][4])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_tHQ_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_tHQ_sample_Other_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_tHQ_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_tHQ_sample_tHQ_categorised.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 4:
                    y_scores_train_tHQ_sample_newphysics_categorised.append(result_probs[i][4])
            
            if Y_train[i][4] == 1:
                wt_train_newphysics_sample.append(train_event_weight)
                y_scores_train_newphysics_sample_ttHnode.append(result_probs[i][0])
                y_scores_train_newphysics_sample_Othernode.append(result_probs[i][1])
                y_scores_train_newphysics_sample_ttWnode.append(result_probs[i][2])
                y_scores_train_newphysics_sample_tHQnode.append(result_probs[i][3])
                y_scores_train_newphysics_sample_newphysicsnode.append(result_probs[i][4])
                if np.argmax(result_probs[i]) == 0:
                    y_scores_train_newphysics_sample_ttH_categorised.append(result_probs[i][0])
                if np.argmax(result_probs[i]) == 1:
                    y_scores_train_newphysics_sample_Other_categorised.append(result_probs[i][1])
                if np.argmax(result_probs[i]) == 2:
                    y_scores_train_newphysics_sample_ttW_categorised.append(result_probs[i][2])
                if np.argmax(result_probs[i]) == 3:
                    y_scores_train_newphysics_sample_tHQ_categorised.append(result_probs[i][3])
                if np.argmax(result_probs[i]) == 4:
                    y_scores_train_newphysics_sample_newphysics_categorised.append(result_probs[i][4])

        wt_test_ttH_sample =[]
        wt_test_Other_sample =[]
        wt_test_ttW_sample =[]
        wt_test_tHQ_sample =[]
        wt_test_newphysics_sample =[]   
        #Arrays to store all ttH values
        y_scores_test_ttH_sample_ttHnode = []
        y_scores_test_Other_sample_ttHnode = []
        y_scores_test_ttW_sample_ttHnode = []
        y_scores_test_tHQ_sample_ttHnode = []
        y_scores_test_newphysics_sample_ttHnode = []
        # Arrays to store ttH categorised event values
        y_scores_test_ttH_sample_ttH_categorised = []
        y_scores_test_Other_sample_ttH_categorised = []
        y_scores_test_ttW_sample_ttH_categorised = []
        y_scores_test_tHQ_sample_ttH_categorised = []
        y_scores_test_newphysics_sample_ttH_categorised = []
        # Arrays to store all Other node values
        y_scores_test_ttH_sample_Othernode = []
        y_scores_test_Other_sample_Othernode = []
        y_scores_test_ttW_sample_Othernode = []
        y_scores_test_tHQ_sample_Othernode = []
        y_scores_test_newphysics_sample_Othernode = []
        # Arrays to store Other categorised event values
        y_scores_test_ttH_sample_Other_categorised = []
        y_scores_test_Other_sample_Other_categorised = []
        y_scores_test_ttW_sample_Other_categorised = []
        y_scores_test_tHQ_sample_Other_categorised = []
        y_scores_test_newphysics_sample_Other_categorised = []
        # Arrays to store all ttW node values
        y_scores_test_ttH_sample_ttWnode = []
        y_scores_test_Other_sample_ttWnode = []
        y_scores_test_ttW_sample_ttWnode = []
        y_scores_test_tHQ_sample_ttWnode = []
        y_scores_test_newphysics_sample_ttWnode = []
        # Arrays to store ttW categorised events
        y_scores_test_ttH_sample_ttW_categorised = []
        y_scores_test_Other_sample_ttW_categorised = []
        y_scores_test_ttW_sample_ttW_categorised = []
        y_scores_test_tHQ_sample_ttW_categorised = []
        y_scores_test_newphysics_sample_ttW_categorised = []
        # Arrays to store all tHQ node values
        y_scores_test_ttH_sample_tHQnode = []
        y_scores_test_Other_sample_tHQnode = []
        y_scores_test_ttW_sample_tHQnode = []
        y_scores_test_tHQ_sample_tHQnode = []
        y_scores_test_newphysics_sample_tHQnode = []
        # Arrays to store tHQ categorised events
        y_scores_test_ttH_sample_tHQ_categorised = []
        y_scores_test_Other_sample_tHQ_categorised = []
        y_scores_test_ttW_sample_tHQ_categorised = []
        y_scores_test_tHQ_sample_tHQ_categorised = []
        y_scores_test_newphysics_sample_tHQ_categorised = []

        # Arrays to store all newphysics node values
        y_scores_test_ttH_sample_newphysicsnode = []
        y_scores_test_Other_sample_newphysicsnode = []
        y_scores_test_ttW_sample_newphysicsnode = []
        y_scores_test_tHQ_sample_newphysicsnode = []
        y_scores_test_newphysics_sample_newphysicsnode = []
        # Arrays to store tHQ categorised events
        y_scores_test_ttH_sample_newphysics_categorised = []
        y_scores_test_Other_sample_newphysics_categorised = []
        y_scores_test_ttW_sample_newphysics_categorised = []
        y_scores_test_tHQ_sample_newphysics_categorised = []
        y_scores_test_newphysics_sample_newphysics_categorised = []
 
        
        for i in range(len(result_probs_test)):
            test_event_weight = test_weights[i]
            if Y_test[i][0] == 1:
                wt_test_ttH_sample.append(test_event_weight)
                y_scores_test_ttH_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_ttH_sample_Othernode.append(result_probs_test[i][1])
                y_scores_test_ttH_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_ttH_sample_tHQnode.append(result_probs_test[i][3])
                y_scores_test_ttH_sample_newphysicsnode.append(result_probs_test[i][4])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_ttH_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_ttH_sample_Other_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_ttH_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_ttH_sample_tHQ_categorised.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 4:
                    y_scores_test_ttH_sample_newphysics_categorised.append(result_probs_test[i][4])

            if Y_test[i][1] == 1:
                wt_test_Other_sample.append(test_event_weight)
                y_scores_test_Other_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_Other_sample_Othernode.append(result_probs_test[i][1])
                y_scores_test_Other_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_Other_sample_tHQnode.append(result_probs_test[i][3])
                y_scores_test_Other_sample_newphysicsnode.append(result_probs_test[i][4])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_Other_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_Other_sample_Other_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_Other_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_Other_sample_tHQ_categorised.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 4:
                    y_scores_test_Other_sample_newphysics_categorised.append(result_probs_test[i][4])
            if Y_test[i][2] == 1:
                wt_test_ttW_sample.append(test_event_weight)
                y_scores_test_ttW_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_ttW_sample_Othernode.append(result_probs_test[i][1])
                y_scores_test_ttW_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_ttW_sample_tHQnode.append(result_probs_test[i][3])
                y_scores_test_ttW_sample_newphysicsnode.append(result_probs_test[i][4])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_ttW_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_ttW_sample_Other_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_ttW_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_ttW_sample_tHQ_categorised.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 4:
                    y_scores_test_ttW_sample_newphysics_categorised.append(result_probs_test[i][4])

            if Y_test[i][3] == 1:
                wt_test_tHQ_sample.append(test_event_weight)
                y_scores_test_tHQ_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_tHQ_sample_Othernode.append(result_probs_test[i][1])
                y_scores_test_tHQ_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_tHQ_sample_tHQnode.append(result_probs_test[i][3])
                y_scores_test_tHQ_sample_newphysicsnode.append(result_probs_test[i][4])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_tHQ_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_tHQ_sample_Other_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_tHQ_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_tHQ_sample_tHQ_categorised.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 4:
                    y_scores_test_tHQ_sample_newphysics_categorised.append(result_probs_test[i][4])

 
            if Y_test[i][4] == 1:
                wt_test_newphysics_sample.append(test_event_weight)
                y_scores_test_newphysics_sample_ttHnode.append(result_probs_test[i][0])
                y_scores_test_newphysics_sample_Othernode.append(result_probs_test[i][1])
                y_scores_test_newphysics_sample_ttWnode.append(result_probs_test[i][2])
                y_scores_test_newphysics_sample_tHQnode.append(result_probs_test[i][3])
                y_scores_test_newphysics_sample_newphysicsnode.append(result_probs_test[i][4])
                if np.argmax(result_probs_test[i]) == 0:
                    y_scores_test_newphysics_sample_ttH_categorised.append(result_probs_test[i][0])
                if np.argmax(result_probs_test[i]) == 1:
                    y_scores_test_newphysics_sample_Other_categorised.append(result_probs_test[i][1])
                if np.argmax(result_probs_test[i]) == 2:
                    y_scores_test_newphysics_sample_ttW_categorised.append(result_probs_test[i][2])
                if np.argmax(result_probs_test[i]) == 3:
                    y_scores_test_newphysics_sample_tHQ_categorised.append(result_probs_test[i][3])
                if np.argmax(result_probs_test[i]) == 4:
                    y_scores_test_newphysics_sample_newphysics_categorised.append(result_probs_test[i][4])
        
        fig, axes = plt.subplots(1, 5, figsize=(50, 10))
        
        alphatrain=0.15
        bins=20
        axes[0].hist(y_scores_train_ttH_sample_ttHnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='r', label="ttH",weights=wt_train_ttH_sample/np.sum(wt_train_ttH_sample),alpha=alphatrain)
        axes[0].hist(y_scores_train_Other_sample_ttHnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='g', label="Other",weights=wt_train_Other_sample/np.sum(wt_train_Other_sample),alpha=alphatrain)
        axes[0].hist(y_scores_train_ttW_sample_ttHnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='c', label="ttW",weights=wt_train_ttW_sample/np.sum(wt_train_ttW_sample),alpha=alphatrain)
        axes[0].hist(y_scores_train_tHQ_sample_ttHnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='k', label="tHQ",weights=wt_train_tHQ_sample/np.sum(wt_train_tHQ_sample),alpha=alphatrain)
        axes[0].hist(y_scores_train_newphysics_sample_ttHnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='m', label="ttHCPodd",weights=wt_train_newphysics_sample/np.sum(wt_train_newphysics_sample),alpha=alphatrain)

        axes[0].hist(y_scores_test_ttH_sample_ttHnode, np.linspace(0, 1, bins), density=False, histtype='step', color='r', label="ttH_test",weights=wt_test_ttH_sample/np.sum(wt_test_ttH_sample),alpha=1,linewidth=4)
        axes[0].hist(y_scores_test_Other_sample_ttHnode, np.linspace(0, 1, bins), density=False, histtype='step', color='g', label="Other_test",weights=wt_test_Other_sample/np.sum(wt_test_Other_sample),alpha=1,linewidth=4)
        axes[0].hist(y_scores_test_ttW_sample_ttHnode, np.linspace(0, 1, bins), density=False, histtype='step', color='c', label="ttW_test",weights=wt_test_ttW_sample/np.sum(wt_test_ttW_sample),alpha=1,linewidth=4)
        axes[0].hist(y_scores_test_tHQ_sample_ttHnode, np.linspace(0, 1, bins), density=False, histtype='step', color='k', label="tHQ_test",weights=wt_test_tHQ_sample/np.sum(wt_test_tHQ_sample),alpha=1,linewidth=4)
        axes[0].hist(y_scores_test_newphysics_sample_ttHnode, np.linspace(0, 1, bins), density=False, histtype='step', color='m', label="ttHCPodd_test",weights=wt_test_newphysics_sample/np.sum(wt_test_newphysics_sample),alpha=1,linewidth=4)

        axes[1].hist(y_scores_train_ttH_sample_Othernode, np.linspace(0, 1, bins), density=False, histtype='bar', color='r', label="ttH",weights=wt_train_ttH_sample/np.sum(wt_train_ttH_sample),alpha=alphatrain)
        axes[1].hist(y_scores_train_Other_sample_Othernode, np.linspace(0, 1, bins), density=False, histtype='bar', color='g', label="Other",weights=wt_train_Other_sample/np.sum(wt_train_Other_sample),alpha=alphatrain)
        axes[1].hist(y_scores_train_ttW_sample_Othernode, np.linspace(0, 1, bins), density=False, histtype='bar', color='c', label="ttW",weights=wt_train_ttW_sample/np.sum(wt_train_ttW_sample),alpha=alphatrain)
        axes[1].hist(y_scores_train_tHQ_sample_Othernode, np.linspace(0, 1, bins), density=False, histtype='bar', color='k', label="tHQ",weights=wt_train_tHQ_sample/np.sum(wt_train_tHQ_sample),alpha=alphatrain)
        axes[1].hist(y_scores_train_newphysics_sample_Othernode, np.linspace(0, 1, bins), density=False, histtype='bar', color='m', label="ttHCPodd",weights=wt_train_newphysics_sample/np.sum(wt_train_newphysics_sample),alpha=alphatrain)

        axes[1].hist(y_scores_test_ttH_sample_Othernode, np.linspace(0, 1, bins), density=False, histtype='step', color='r', label="ttH_test",weights=wt_test_ttH_sample/np.sum(wt_test_ttH_sample),alpha=1,linewidth=4)
        axes[1].hist(y_scores_test_Other_sample_Othernode, np.linspace(0, 1, bins), density=False, histtype='step', color='g', label="Other_test",weights=wt_test_Other_sample/np.sum(wt_test_Other_sample),alpha=1,linewidth=4)
        axes[1].hist(y_scores_test_ttW_sample_Othernode, np.linspace(0, 1, bins), density=False, histtype='step', color='c', label="ttW_test",weights=wt_test_ttW_sample/np.sum(wt_test_ttW_sample),alpha=1,linewidth=4)
        axes[1].hist(y_scores_test_tHQ_sample_Othernode, np.linspace(0, 1, bins), density=False, histtype='step', color='k', label="tHQ_test",weights=wt_test_tHQ_sample/np.sum(wt_test_tHQ_sample),alpha=1,linewidth=4)
        axes[1].hist(y_scores_test_newphysics_sample_Othernode, np.linspace(0, 1, bins), density=False, histtype='step', color='m', label="ttHCPodd_test",weights=wt_test_newphysics_sample/np.sum(wt_test_newphysics_sample),alpha=1,linewidth=4)

        axes[2].hist(y_scores_train_ttH_sample_ttWnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='r', label="ttH",weights=wt_train_ttH_sample/np.sum(wt_train_ttH_sample),alpha=alphatrain)
        axes[2].hist(y_scores_train_Other_sample_ttWnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='g', label="Other",weights=wt_train_Other_sample/np.sum(wt_train_Other_sample),alpha=alphatrain)
        axes[2].hist(y_scores_train_ttW_sample_ttWnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='c', label="ttW",weights=wt_train_ttW_sample/np.sum(wt_train_ttW_sample),alpha=alphatrain)
        axes[2].hist(y_scores_train_tHQ_sample_ttWnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='k', label="tHQ",weights=wt_train_tHQ_sample/np.sum(wt_train_tHQ_sample),alpha=alphatrain)
        axes[2].hist(y_scores_train_newphysics_sample_ttWnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='m', label="ttHCPodd",weights=wt_train_newphysics_sample/np.sum(wt_train_newphysics_sample),alpha=alphatrain)

        axes[2].hist(y_scores_test_ttH_sample_ttWnode, np.linspace(0, 1, bins), density=False, histtype='step', color='r', label="ttH_test",weights=wt_test_ttH_sample/np.sum(wt_test_ttH_sample),alpha=1,linewidth=4)
        axes[2].hist(y_scores_test_Other_sample_ttWnode, np.linspace(0, 1, bins), density=False, histtype='step', color='g', label="Other_test",weights=wt_test_Other_sample/np.sum(wt_test_Other_sample),alpha=1,linewidth=4)
        axes[2].hist(y_scores_test_ttW_sample_ttWnode, np.linspace(0, 1, bins), density=False, histtype='step', color='c', label="ttW_test",weights=wt_test_ttW_sample/np.sum(wt_test_ttW_sample),alpha=1,linewidth=4)
        axes[2].hist(y_scores_test_tHQ_sample_ttWnode, np.linspace(0, 1, bins), density=False, histtype='step', color='k', label="tHQ_test",weights=wt_test_tHQ_sample/np.sum(wt_test_tHQ_sample),alpha=1,linewidth=4)
        axes[2].hist(y_scores_test_newphysics_sample_ttWnode, np.linspace(0, 1, bins), density=False, histtype='step', color='m', label="ttHCPodd_test",weights=wt_test_newphysics_sample/np.sum(wt_test_newphysics_sample),alpha=1,linewidth=4)

        axes[3].hist(y_scores_train_ttH_sample_tHQnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='r', label="ttH",weights=wt_train_ttH_sample/np.sum(wt_train_ttH_sample),alpha=alphatrain)
        axes[3].hist(y_scores_train_Other_sample_tHQnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='g', label="Other",weights=wt_train_Other_sample/np.sum(wt_train_Other_sample),alpha=alphatrain)
        axes[3].hist(y_scores_train_ttW_sample_tHQnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='c', label="ttW",weights=wt_train_ttW_sample/np.sum(wt_train_ttW_sample),alpha=alphatrain)
        axes[3].hist(y_scores_train_tHQ_sample_tHQnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='k', label="tHQ",weights=wt_train_tHQ_sample/np.sum(wt_train_tHQ_sample),alpha=alphatrain)
        axes[3].hist(y_scores_train_newphysics_sample_tHQnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='m', label="ttHCPodd",weights=wt_train_newphysics_sample/np.sum(wt_train_newphysics_sample),alpha=alphatrain)

        axes[3].hist(y_scores_test_ttH_sample_tHQnode, np.linspace(0, 1, bins), density=False, histtype='step', color='r', label="ttH_test",weights=wt_test_ttH_sample/np.sum(wt_test_ttH_sample),alpha=1,linewidth=4)
        axes[3].hist(y_scores_test_Other_sample_tHQnode, np.linspace(0, 1, bins), density=False, histtype='step', color='g', label="Other_test",weights=wt_test_Other_sample/np.sum(wt_test_Other_sample),alpha=1,linewidth=4)
        axes[3].hist(y_scores_test_ttW_sample_tHQnode, np.linspace(0, 1, bins), density=False, histtype='step', color='c', label="ttW_test",weights=wt_test_ttW_sample/np.sum(wt_test_ttW_sample),alpha=1,linewidth=4)
        axes[3].hist(y_scores_test_tHQ_sample_tHQnode, np.linspace(0, 1, bins), density=False, histtype='step', color='k', label="tHQ_test",weights=wt_test_tHQ_sample/np.sum(wt_test_tHQ_sample),alpha=1,linewidth=4)
        axes[3].hist(y_scores_test_newphysics_sample_tHQnode, np.linspace(0, 1, bins), density=False, histtype='step', color='m', label="ttHCPodd_test",weights=wt_test_newphysics_sample/np.sum(wt_test_newphysics_sample),alpha=1,linewidth=4)

        axes[4].hist(y_scores_train_ttH_sample_newphysicsnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='r', label="ttH",weights=wt_train_ttH_sample/np.sum(wt_train_ttH_sample),alpha=alphatrain)
        axes[4].hist(y_scores_train_Other_sample_newphysicsnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='g', label="Other",weights=wt_train_Other_sample/np.sum(wt_train_Other_sample),alpha=alphatrain)
        axes[4].hist(y_scores_train_ttW_sample_newphysicsnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='c', label="ttW",weights=wt_train_ttW_sample/np.sum(wt_train_ttW_sample),alpha=alphatrain)
        axes[4].hist(y_scores_train_tHQ_sample_newphysicsnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='k', label="tHQ",weights=wt_train_tHQ_sample/np.sum(wt_train_tHQ_sample),alpha=alphatrain)
        axes[4].hist(y_scores_train_newphysics_sample_newphysicsnode, np.linspace(0, 1, bins), density=False, histtype='bar', color='m', label="ttHCPodd",weights=wt_train_newphysics_sample/np.sum(wt_train_newphysics_sample),alpha=alphatrain)

        axes[4].hist(y_scores_test_ttH_sample_newphysicsnode, np.linspace(0, 1, bins), density=False, histtype='step', color='r', label="ttH_test",weights=wt_test_ttH_sample/np.sum(wt_test_ttH_sample),alpha=1,linewidth=4)
        axes[4].hist(y_scores_test_Other_sample_newphysicsnode, np.linspace(0, 1, bins), density=False, histtype='step', color='g', label="Other_test",weights=wt_test_Other_sample/np.sum(wt_test_Other_sample),alpha=1,linewidth=4)
        axes[4].hist(y_scores_test_ttW_sample_newphysicsnode, np.linspace(0, 1, bins), density=False, histtype='step', color='c', label="ttW_test",weights=wt_test_ttW_sample/np.sum(wt_test_ttW_sample),alpha=1,linewidth=4)
        axes[4].hist(y_scores_test_tHQ_sample_newphysicsnode, np.linspace(0, 1, bins), density=False, histtype='step', color='k', label="tHQ_test",weights=wt_test_tHQ_sample/np.sum(wt_test_tHQ_sample),alpha=1,linewidth=4)
        axes[4].hist(y_scores_test_newphysics_sample_newphysicsnode, np.linspace(0, 1, bins), density=False, histtype='step', color='m', label="ttHCPodd_test",weights=wt_test_newphysics_sample/np.sum(wt_test_newphysics_sample),alpha=1,linewidth=4)

        
        axes[0].legend(prop={'size': 20},loc='upper right')
        axes[0].set_ylabel('Events')
        axes[0].set_xlabel('Output')
        
        axes[1].legend(prop={'size': 20},loc='upper right')
        axes[1].set_ylabel('Events')
        axes[1].set_xlabel('Output')
        
        axes[2].legend(prop={'size': 20},loc='upper right')
        axes[2].set_ylabel('Events')
        axes[2].set_xlabel('Output')
        
        axes[3].legend(prop={'size': 20},loc='upper right')
        axes[3].set_ylabel('Events')
        axes[3].set_xlabel('Output')
        
        axes[4].legend(prop={'size': 20},loc='upper right')
        axes[4].set_ylabel('Events')
        axes[4].set_xlabel('Output')
        
        axes[0].set_ylim(0, 0.7)
        axes[1].set_ylim(0, 0.7)
        axes[2].set_ylim(0, 0.7)
        axes[3].set_ylim(0, 0.7)
        axes[4].set_ylim(0, 0.7)
        
        axes[0].set_title("ttH Node",fontsize=40)
        axes[1].set_title("Other Node",fontsize=40)
        axes[2].set_title("ttW Node",fontsize=40)
        axes[3].set_title("tHQ Node",fontsize=40)
        axes[4].set_title("ttHCPodd Node",fontsize=40)
        
        axes[0].set_xlim(0, 0.6)
        axes[1].set_xlim(0, 0.6)
        axes[2].set_xlim(0, 0.6)
        axes[3].set_xlim(0, 0.6)
        axes[4].set_xlim(0, 0.6)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig('result/'+ timestr +'plot.pdf')



        fpr = dict()
        tpr = dict()
        fpr_training=dict()
        tpr_training=dict()
        roc_auc = dict()
        roc_auc_training=dict()
   
        for i in range(5):
            fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], result_probs_test[:, i],sample_weight=test_weights)
            fpr_training[i],tpr_training[i],_training = roc_curve(Y_train[:, i], result_probs[:, i],sample_weight=train_weights)
            roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc_training[i]=auc(fpr_training[i],tpr_training[i])
        fig, ax = plt.subplots(5,figsize=(10, 50)) 
        for i in range(5):
            ax[i].plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            ax[i].plot(fpr_training[i], tpr_training[i], label='ROC curve training (area = %0.2f)' % roc_auc_training[i])
            ax[i].plot([0, 1], [0, 1], 'k--')
            ax[i].set_xlim([0.0, 1.0])
            ax[i].set_ylim([0.0, 1.05])
            ax[i].set_xlabel('False Positive Rate')
            ax[i].set_ylabel('True Positive Rate')
            ax[i].set_title('ROC node '+ str(i+1))
            ax[i].legend(loc="lower right")
        plt.savefig('result/'+ timestr+ 'roc.pdf')
    

        






