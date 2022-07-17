#############################
# import necessary packages #
#############################
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib. pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.datasets import load_breast_cancer

from dwave.system import DWaveSampler, EmbeddingComposite
from qboost import QBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from timeit import default_timer as timer
from collections import Counter
from sklearn.metrics import confusion_matrix
from prepare_datasets import scompatta_dataset
from sklearn.metrics import classification_report
from sys import exit


#######################
# Settings parameters #
#######################
DEVtoken = 'DEV-48223ef7d367a60424a58b83f37d5bc879ba189d'
percentage = 0.20 #percentage of dataset

##################
# Choose dataset #
##################
'''
Uncomment ONE of these datasets to use it
'''
dataset = "bow_dir"
# dataset = "tf_dir"
# dataset = "tfidf_dir"

#######################
# Functions #
#######################

def convert_labels(y_data):
    """
    :param y_data: labels 
    :return: array of numerical labels (0:non-malicious,1:malicious)
    """
    y_data = y_data.values.ravel()
    y_data = np.where(y_data, 1, -1)

    return y_data
    

def remove_multiclass(df):
    df = df[df['label'].isin([0, 1])] 
    df = df.sample(frac=percentage)
    return df

def transform_dependent_values(y_data):
    y_data = y_data.values.ravel()
    y_data = np.where(y_data, 1, -1)

    return y_data

################
# Load dataset #
################
choosen = scompatta_dataset(dataset)
print('Dataset loaded')

# Removing multiclass dataset #
###############################
# choosen = remove_multiclass(choosen)

# label_list = choosen['label'].to_numpy()
# print(np.unique(label_list))
# print(choosen['EDB_exploitable'])
# exit(1)
#######################################
# Removing label from dataset columns #
#######################################
print('splitting dataset... ')
cols = list(choosen.columns)
# cols.remove('label')
cols.remove('EDB_exploitable')
choosen['EDB_exploitable'] = choosen['EDB_exploitable'].astype(int)

#####################################
# Split columns for indipendentList #
#####################################
x_train,x_test,y_train,y_test=train_test_split(choosen[cols],choosen[['EDB_exploitable']],test_size=0.2,shuffle=True)

# print(y_train)
# exit(1)
############################
# Convert labels into 1/-1 #
############################
y_train = convert_labels(y_train)
y_test = convert_labels(y_test)


################################################
# Create svc model and fit it on training data #
################################################
print('training svc model... ')
svc = SVC(kernel='linear', C = 1.0)
start = timer()
svc.fit(x_train, y_train)
end = timer()
print(f'SVC training time in seconds: {(end - start)}')
tempo = (end - start)

####################################
# Predict accuracy on training set #
####################################
print('predicting on training set svc...')
predictions = svc.predict(x_train)
print("Reporting...")
report = classification_report(y_train, predictions)
with open('report/report_' + dataset + '.txt', 'a') as f:
    f.write(report)
    f.write('tempo normale training: '+str(tempo)+'\n')
f.close()
print("saved...")

##################################
# Predict results on testing set #
##################################
print('predicting svc... ')
start = timer()
predictions = svc.predict(x_test)
end = timer()
tempo = (end - start)
print("Reporting...")
report = classification_report(y_test, predictions)
with open('report/report_' + dataset + '.txt', 'a') as f:
    f.write(report)
    f.write('tempo normale testing: '+str(tempo)+'\n')
f.close()
print("saved...")

##################
# Define sampler #
##################
dwave_sampler = DWaveSampler()
emb_sampler = EmbeddingComposite(dwave_sampler)
lmd = 0.04

######################
# Train Qboost model #
######################
print('starting qboost training...')
#parametri di classificazione di qboost
DW_PARAMS = {'num_reads': 2,
                'auto_scale': True,
                # "answer_mode": "histogram",
                'num_spin_reversal_transforms': 2,
                # 'annealing_time': 10,
                #'postprocess': 'optimization',
                }
NUM_WEAK_CLASSIFIERS = 2
TREE_DEPTH = 1
dwave_sampler = DWaveSampler(token=DEVtoken)
# sa_sampler = micro.dimod.SimulatedAnnealingSampler()
emb_sampler = EmbeddingComposite(dwave_sampler)
lmd = 0.5
#fine paramentri
start = timer()
qboost = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
qboost.fit(x_train, y_train, emb_sampler, lmd=lmd, **DW_PARAMS)
end = timer()
print(f'QBoost training time in seconds: {(end - start)}')
tempo = (end - start)


####################################
# Predict accuracy on training set #
####################################
print('predicting on training set qboost...')
predictions = qboost.predict(x_train) 
report = classification_report(y_train, predictions)
# report/report.txt
with open('report/report_' + dataset + '.txt', 'a') as f:
    f.write(report)
    f.write('tempo qboost training: '+str(tempo)+'\n')
f.close()
print("saved...")

####################################
# Predict with Qboost on test data #
####################################
print('predicting qboost... ')
start = timer()
predictions = qboost.predict(x_test)
end = timer()
tempo = (end - start)
print(f'QBoost prediction time in seconds: {(end - start)}')
report = classification_report(y_test, predictions)
with open('report/report_' + dataset + '.txt', 'a') as f:
    f.write(report)
    f.write('tempo qboost testing: '+str(tempo)+'\n')
f.close()
print("saved...")
