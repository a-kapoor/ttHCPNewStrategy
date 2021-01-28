#!/usr/bin/env python
# coding: utf-8

# In[26]:


import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import optparse, json, argparse, math
import ROOT
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
import os
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Reshape,Conv2D,MaxPooling2D,ConvLSTM2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
import newhelper as newhelper
from root_numpy import root2array, tree2array
seed = 7
##small changes
np.random.seed(7)
rng = np.random.RandomState(31337)
timestr=time.strftime("%Y%m%d-%H%M%S")
def load_data(inputPath,variables,criteria):
    # Load dataset to .csv format file
    my_cols_list=variables+['process', 'key', 'target', 'totalWeight','sampleWeight']
    data = pd.DataFrame(columns=my_cols_list)
    keys=['ttH','ttJ','ttW','THQ','newphysics']

    for key in keys :
        print(key)
        if 'ttH' in key or 'TTH' in key:
            #data["EVENT_originalXWGTUP"]=1
            sampleNames=['ttH']          
            #fileNames = ['ttHnobb_DiLepRegion']
            fileNames = ['TTH_ctcvcp_DiLepRegion_New2017']
            target=0
        if 'ttJ' in key:
            sampleNames=['ttJ']
            fileNames = ['ttJets_PS_DiLepRegion']
            target=1
        if 'ttW' in key or 'TTW' in key:
            sampleNames=['ttW']
            fileNames=['ttWJets_DiLepRegion']
            target=2
        if 'THQ' in key:
            sampleNames=['THQ']
            fileNames=['THQ_ctcvcp_DiLepRegion']
            #criteria = "(" + criteria + "&&(nEvent%3==0))"
            target=3
        if 'newphysics' in key:
            #data["EVENT_originalXWGTUP"]=1
            sampleNames=['newphysics']
            fileNames = ['TTH_ctcvcp_DiLepRegion_New2017']
            
            #fileNames = ['ttHnobb_DiLepRegion']
            target=4
        inputTree = 'syncTree'
        print(sampleNames)

        for process_index in range(len(fileNames)):
            fileName = fileNames[process_index]
            sampleName = sampleNames[process_index]

            try: tfile = ROOT.TFile(inputPath+"/"+fileName+".root")
            except :
                print(" file "+ inputPath+"/"+fileName+".root doesn't exits ")
                continue
            try: tree = tfile.Get(inputTree)
            except :
                print(inputTree + " deosn't exists in " + inputPath+"/"+fileName+".root")
                continue
            if tree is not None :
                print('criteria: ', criteria)
                #try: chunk_arr = tree2array(tree=tree, selection=criteria, start=0, stop=100) # Can use  start=first entry, stop = final entry desired
                try: chunk_arr = tree2array(tree=tree, selection=criteria) # Can use  start=first entry, stop = final entry desired
                except : continue
                else :
                    chunk_df = pd.DataFrame(chunk_arr, columns=variables)
                    chunk_df['process']=sampleName
                    chunk_df['key']=key
                    chunk_df['target']=target
                    chunk_df['totalWeight']=chunk_df["EventWeight"]
                    chunk_df['nEvent']=chunk_df["nEvent"]

                    if sampleName=='ttH':
                        chunk_df['CPWeighto'] = [x[11] for x in chunk_df['EVENT_rWeights']]
                        chunk_df['sampleWeight'] = chunk_df['CPWeighto']#/chunk_df['EVENT_originalXWGTUP']
                        chunk_df['sampleWeight']= chunk_df['sampleWeight']/(chunk_df['sampleWeight'].sum())
                        print('ttH wt sum:'+str(chunk_df['sampleWeight'].sum()))                                                         
                    if sampleName=='ttJ':
                        chunk_df['sampleWeight'] = 1.0
                        chunk_df['sampleWeight']= chunk_df['sampleWeight']/(chunk_df['sampleWeight'].sum())
                        print('ttJ wt sum:'+str(chunk_df['sampleWeight'].sum()))
                    if sampleName=='ttW':
                        chunk_df['sampleWeight'] = 1#0.601
                        chunk_df['sampleWeight']= chunk_df['sampleWeight']/(chunk_df['sampleWeight'].sum())
                        print('ttW wt sum:'+str(chunk_df['sampleWeight'].sum()))
                    if sampleName=='THQ': 
                        chunk_df['sampleWeight'] = 1#0.0231
                        chunk_df['sampleWeight']= chunk_df['sampleWeight']/(chunk_df['sampleWeight'].sum())
                        #print('tHQ wt sum:'+str(chunk_df['sampleWeight'].sum()))
                        print('tHQ wt sum:'+str(chunk_df['sampleWeight'].sum()))
                    if sampleName=='newphysics':
                        chunk_df['CPWeighto'] = [x[59] for x in chunk_df['EVENT_rWeights']]
                        chunk_df['sampleWeight'] = chunk_df['CPWeighto']#/chunk_df['EVENT_originalXWGTUP'] 
                        chunk_df['sampleWeight']= chunk_df['sampleWeight']/(chunk_df['sampleWeight'].sum())
                        print('newPhysics wt sum:'+str(chunk_df['sampleWeight'].sum()))

                    #chunk_df[['jet1_eta','jet2_eta','jet3_eta','jet4_eta','jetFwd1_eta']] = chunk_df[['jet1_eta','jet2_eta','jet3_eta','jet4_eta','jetFwd1_eta']].apply(np.absolute)
                    data = data.append(chunk_df, ignore_index=True)
            tfile.Close()
        if len(data) == 0 : continue
        nttH = len(data.iloc[(data.target.values == 0) & (data.key.values==key) ])
        nttJ = len(data.iloc[(data.target.values == 1) & (data.key.values==key) ])
        nttW = len(data.iloc[(data.target.values == 2) & (data.key.values==key) ])
        nTHQ = len(data.iloc[(data.target.values == 3) & (data.key.values==key) ])
        nnewphysics = len(data.iloc[(data.target.values == 4) & (data.key.values==key) ])
        nOther00 = len(data.iloc[(data.target.values == 5) & (data.key.values==key) ])
        processfreq = data.groupby('key')
        samplefreq = data.groupby('process')

        if key == 'ttH':
            print('Process ttH frequency: ', len(processfreq.get_group('ttH')))
        elif key == 'ttJ':
            print('Target Other frequency: ', len(samplefreq.get_group('ttJ')))
        elif key == 'ttW':
            print('Process ttW frequency: ', len(processfreq.get_group('ttW')))
        elif key == 'THQ':
            print('Process tHQ frequency: ', len(processfreq.get_group('THQ')))
        elif key == 'newphysics':
            print('Process newphysics frequency: ', len(processfreq.get_group('newphysics')))

        print("TotalWeights = %f" % (data.iloc[(data.key.values==key)]["totalWeight"].sum()))
        nNW = len(data.iloc[(data["totalWeight"].values < 0) & (data.key.values==key) ])
        print(key, "events with -ve weights", nNW)
    print('<load_data> data columns: ', (data.columns.values.tolist()))
    n = len(data)
    nttH = len(data.iloc[data.target.values == 0])
    nttJ = len(data.iloc[data.target.values == 1])
    nttW = len(data.iloc[data.target.values == 2])
    nTHQ = len(data.iloc[data.target.values == 3])
    nnewphysics = len(data.iloc[data.target.values == 4])
    nOther = len(data.iloc[data.target.values == 5])
    print("Total length of nttH = %i, ttJ = %i, nttW = %i, nTHQ = %i, nnewphysics = %i" % (nttH, nttJ , nttW, nTHQ, nnewphysics))
    data.fillna(0)
    return data

def load_trained_model(weights_path, num_variables, optimizer,nClasses):
    model = baseline_model(num_variables, optimizer,nClasses)
    model.load_weights(weights_path)
    return model

def normalise(x_train, x_test):
    mu = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_normalised = (x_train - mu) / std
    x_test_normalised = (x_test - mu) / std
    return x_train_normalised, x_test_normalised

def baseline_model(num_variables,optimizer,nClasses):
    model = Sequential()
    model.add(Dense(32,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    for index in range(3):
        model.add(Dense(16,activation='relu'))
        model.add(Dropout(0.1))
    for index in range(2):
        model.add(Dense(16,activation='relu'))
    for index in range(2):
        model.add(Dense(8,activation='relu'))
        model.add(Dropout(0.1))
    model.add(Dense(nClasses, activation='softmax'))
    if optimizer=='Adam':
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['acc'])
    if optimizer=='Nadam':
        model.compile(loss='categorical_crossentropy',optimizer=Nadam(lr=0.001),metrics=['acc'])
    return model

# def baseline_model(num_variables,optimizer,nClasses):
#     model = Sequential()
#     model.add(Dense(36,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
#     for index in range(2):
#         model.add(Dense(64,activation='relu'))
#         model.add(Dropout(0.1))
#     for index in range(2):
#         model.add(Dense(32,activation='relu'))
#     for index in range(2):
#         model.add(Dense(16,activation='relu'))
#     model.add(Dense(nClasses, activation='softmax'))
#     if optimizer=='Adam':
#         model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['acc'])
#     if optimizer=='Nadam':
#         model.compile(loss='categorical_crossentropy',optimizer=Nadam(lr=0.001),metrics=['acc'])
#     return model

def newCNN_model(num_variables,optimizer,nClasses,nl,dp):
    model = Sequential()
    model.add(Dense(36,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Reshape((1,6, 6),input_shape=(36,)))
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    #model.add(Conv2D(4, (3, 3), padding='same',activation='relu'))
    #model.add(Conv2D(8, (3, 3), padding='same',activation='relu'))
    #model.add(Conv2D(8, (3, 3), padding='same',activation='relu'))
    model.add(Flatten())
    model.add(Dense(36,input_dim=num_variables,activation='relu'))
    for index in range(5):
        model.add(Dense(16,activation='relu'))
        model.add(Dropout(0.1))
    for index in range(5):
        model.add(Dense(16,activation='relu'))
    for index in range(5):
        model.add(Dense(8,activation='relu'))
    #model.add(Dense(100,activation='relu'))
    #model.add(Dense(25,activation='relu'))
    #                                                                                                                                                                                                                                                                           
    # for index in range(5):                                                                                                                                                                                                                                                    
    #     model.add(Dense(16,activation='relu'))                                                                                                                                                                                                                                
    #     model.add(Dropout(0.01))                                                                                                                                                                                                                                              
    # for index in range(5):                                                                                                                                                                                                                                                    
    #     model.add(Dense(16,activation='relu'))                                                                                                                                                                                                                                
    # for index in range(5):                                                                                                                                                                                                                                                    
    #     model.add(Dense(8,activation='relu'))                                                                                                                                                                                                                                 
    model.add(Dense(nClasses, activation='softmax'))
    if optimizer=='Adam':
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['acc'])
    if optimizer=='Nadam':
        model.compile(loss='categorical_crossentropy',optimizer=Nadam(lr=0.001),metrics=['acc'])
    if optimizer=='RMSprop':
        model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['acc'])
    return model



def newCNN_model_new(num_variables,optimizer,nClasses,nl,dp):
    model = Sequential()
    model.add(Dense(36,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Reshape((1,6, 6),input_shape=(36,)))
    #model.add(Conv2D(256, (3, 3), padding='same',activation='relu'))                                                                                                                                                                                                           
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same',activation='relu'))                                                                                                                                                                                                            
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(25,activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))
    if optimizer=='Adam':
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['acc'])
    if optimizer=='Nadam':
        model.compile(loss='categorical_crossentropy',optimizer=Nadam(lr=0.001),metrics=['acc'])
    if optimizer=='RMSprop':
        model.compile(loss='categorical_crossentropy',optimizer=RMSprop(learning_rate=0.001),metrics=['acc'])
    return model


def check_dir(dir):
    if not os.path.exists(dir):
        print('mkdir: ', dir)
        os.makedirs(dir)

# Ratio always > 1. mu use in natural log multiplied into ratio. Keep mu above 1 to avoid class weights going negative.
def create_class_weight(labels_dict,mu=0.9):
    total = np.sum(list(labels_dict.values())) # total number of examples in all datasets
    keys = list(labels_dict.keys()) # labels
    class_weight = dict()
    print('total: ', total)

    for key in keys:
        # logarithm smooths the weights for very imbalanced classes.
        score = math.log(mu*total/float(labels_dict[key])) # natlog(parameter * total number of examples / number of examples for given label)
        #score = float(total/labels_dict[key])
        print('score = ', score)
        if score > 0.:
            class_weight[key] = score
        else :
            class_weight[key] = 1.
    return class_weight


# In[27]:



print('Using Keras version: ', tf.keras.__version__)

do_model_fit = 1
classweights_name = 'InverseSRYields'#args.classweights''
selection = 'th'#args.selection''
# Number of classes to use
number_of_classes = 5
# Create instance of output directory where all results are saved.
#output_directory = '/publicfs/cms/data/TopQuark/cms13TeV/ForDNNSharing/result'
output_directory = '/publicfs/cms/user/yuancc/DarkMatter/MUSICsim/ttHCPNewStrategy/ttH_multilepton/keras-DNN/result'
    
check_dir(output_directory)

# Create plots subdirectory
plots_dir = os.path.join(output_directory,'plots/'+timestr)
plots_dir3 = os.path.join(output_directory,'plots3/')
plots_dir4 = os.path.join(output_directory,'plots4/')

#input_var_jsonFile = open('input_vars_SigRegion_wFwdJet.json','r')
input_var_jsonFile = open('input_vars_SigRegion_wFwdJet.json','r')

    
selection_criteria = '(is_tH_like_and_not_ttH_like==0 || is_tH_like_and_not_ttH_like==1) && Entry$%10==0'#&& n_presel_jet>=3'

# Load Variables from .json
variable_list = list(json.load(input_var_jsonFile,encoding="utf-8").items())

# Create list of headers for dataset .csv
column_headers = []
for key,var in variable_list:
    column_headers.append(key)
column_headers.append('EventWeight')
column_headers.append('xsec_rwgt')
column_headers.append('nEvent')
column_headers.append('EVENT_rWeights')

# Create instance of the input files directory
inputs_file_path = '/publicfs/cms/data/TopQuark/cms13TeV/ForDNNSharing/sample'

# Load ttree into .csv including all variables listed in column_headers
print('<train-DNN> Input file path: ', inputs_file_path)
outputdataframe_name = '%s/output_dataframe_%s.csv' %(output_directory,selection)
if os.path.isfile(outputdataframe_name):
    data = pandas.read_csv(outputdataframe_name)
    data.fillna(0)
    print('<train-DNN> Loading data .csv from: %s . . . . ' % (outputdataframe_name))
else:
    print('<train-DNN> Creating new data .csv @: %s . . . . ' % (inputs_file_path))
    data = load_data(inputs_file_path,column_headers,selection_criteria)
    data.to_csv(outputdataframe_name, index=False)
    data = pandas.read_csv(outputdataframe_name)

# Make instance of plotter tool
#Plotter = plotter()

# Create statistically independant lists train/test data (used to train/evaluate the network)
traindataset, valdataset = train_test_split(data, test_size=0.5)
#valdataset.to_csv('valid_dataset.csv', index=False)
training_columns = column_headers[:-3]
print('<train-DNN> Training features: ', training_columns)

# Select data from columns under the remaining column headers in traindataset
X_train = traindataset[training_columns].values
Y_train = traindataset.target.astype(int)
X_test = valdataset[training_columns].values
Y_test = valdataset.target.astype(int)

num_variables = len(training_columns)

    # Create dataframe containing input features only (for correlation matrix)
train_df = data.iloc[:traindataset.shape[0]]
train_df.drop(['EventWeight'], axis=1, inplace=True)
train_df.drop(['xsec_rwgt'], axis=1, inplace=True)

## Input Variable Correlation plot
#correlation_plot_file_name = 'correlation_plot.png'

#sampleweights = traindataset.loc[:,'sampleWeight']*traindataset.loc[:,'xsec_rwgt']
sampleweights = traindataset.loc[:,'sampleWeight']
sampleweights = np.array(sampleweights)

#train_weights = traindataset['xsec_rwgt'].values
#test_weights = valdataset['xsec_rwgt'].values
train_weights = traindataset['sampleWeight'].values
test_weights = valdataset['sampleWeight'].values
# Fit label encoder to Y_train
newencoder = LabelEncoder()
newencoder.fit(Y_train)
# Transform to encoded array
encoded_Y = newencoder.transform(Y_train)
encoded_Y_test = newencoder.transform(Y_test)
# Transform to one hot encoded arrays
# Y_train = np_utils.to_categorical(encoded_Y)
# Y_test = np_utils.to_categorical(encoded_Y_test)
Y_train = to_categorical(encoded_Y)
Y_test = to_categorical(encoded_Y_test)
optimizer = 'Adam'#'Nadam'
if do_model_fit == 1:
    histories = []
    labels = []
    # Define model and early stopping                                                                                                         
    early_stopping_monitor = EarlyStopping(patience=150,monitor='val_loss',verbose=1)
    model3 = baseline_model(num_variables,optimizer,number_of_classes)
    #model3 = newCNN_model(num_variables,optimizer,number_of_classes,1000,0.40)
    history3 = model3.fit(X_train,Y_train,validation_split=0.2,epochs=100,batch_size=1000,verbose=1,shuffle=True,sample_weight=sampleweights,callbacks=[early_stopping_monitor])
    histories.append(history3)
    labels.append(optimizer)

    # Make plot of loss function evolution                                                                                               
    #Plotter.plot_training_progress_acc(histories, labels)
    #acc_progress_filename = 'DNN_acc_wrt_epoch.pdf'
    #Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename)
     #Which model do you want the rest of the plots for?                                                                                         
    model = model3
else:
    # Which model do you want to load?                  
    model_name = os.path.join(output_directory,'model.h5')
    print('<train-DNN> Loaded Model: %s' % (model_name))
    model = load_trained_model(model_name,num_variables,optimizer,number_of_classes)
# Node probabilities for training sample events
result_probs = model.predict(np.array(X_train))
result_classes = model.predict_classes(np.array(X_train))

# Node probabilities for testing sample events     
result_probs_test = model.predict(np.array(X_test))
result_classes_test = model.predict_classes(np.array(X_test))
# Store model in file                                        
model_output_name = os.path.join(output_directory,'model.h5')
model.save(model_output_name)
weights_output_name = os.path.join(output_directory,'model_weights.h5')
model.save_weights(weights_output_name)
model_json = model.to_json()
model_json_name = os.path.join(output_directory,'model_serialised.json')
with open(model_json_name,'w') as json_file:
    json_file.write(model_json)
model.summary()
model_schematic_name = os.path.join(output_directory,'model_schematic.pdf')
#plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True)

# Initialise output directory.     
#Plotter.plots_directory = plots_dir
#Plotter.output_directory = output_directory
    


# In[28]:


# Make overfitting plots of output nodes
                          


# In[66]:


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
        
        #plt.savefig('result/plot.pdf')      
        


# In[67]:
newhelper.newoverfitting(model, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights)


# In[ ]:





# In[ ]:




