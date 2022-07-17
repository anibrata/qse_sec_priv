#!/usr/bin/env python
# coding: utf-8

#############################
# import necessary packages #
#############################

import os
import time
import pandas as pd
import numpy as nm
import numpy as np
from scipy.sparse import load_npz

import seaborn as sns
import matplotlib. pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.datasets import load_breast_cancer
from dwave.system import DWaveSampler, EmbeddingComposite
from qboost import QBoostClassifier

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from timeit import default_timer as timer
from collections import Counter
from sklearn.metrics import confusion_matrix
#from prepare_datasets import scompatta_dataset
from sklearn.metrics import classification_report

from sklearn import linear_model

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from datetime import datetime
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier

from sys import exit

#######################
# Settings parameters #
#######################
DEVtoken = 'DEV-98f903479d1e03bc59d7ba92376a492f76f7c906'

''' 
Prepare dataset to be read from the raw data & load into dataframe
'''
def load_dataframe(dir):
    data_filepath = os.path.join(dir, "data.npz")
    index_filepath = os.path.join(dir, "indexes.txt")
    cols_filepath = os.path.join(dir, "cols.txt")
    coo = load_npz(data_filepath)
    with open(index_filepath) as file:
        indexes = file.read().splitlines()
    with open(cols_filepath) as file:
        cols = file.read().splitlines()
    return pd.DataFrame.sparse.from_spmatrix(coo, index=indexes, columns=cols)


''' 
Load features into and create joins
'''
def load_features(left_df, dir):
    # Remember that this is made of SparseDType columns
    right_df = load_dataframe(dir)
    right_df.index.name = "cve"
    print(f"Joining Independent Variables")
    full_df = left_df.join(right_df, how="left")
    return full_df


''' 
Prepare dataset for proper dataframe and machine learning
'''
def prepare_dataset_for_ml(dir):
    nvd_filepath = "data/cves/cves.csv"
    exploits_filepath = "data/exploits/cves_exploits.csv"

    nvd_df = pd.read_csv(nvd_filepath, sep=";")
    nvd_df.drop(["pub_date", "description", "references", "cwe"], axis=1, inplace=True)
    nvd_df.set_index("cve", inplace=True)
    #print('Columns :', nvd_df.index)
    nvd_df.rename({col: "NVD_" + col for col in nvd_df.columns}, axis=1, inplace=True)
    #print('data frame :\n', nvd_df)

    #exit()

    exploits_df = pd.read_csv(exploits_filepath)
    #print(exploits_df.columns.tolist())
    exploits_df.drop([col for col in exploits_df.columns.tolist() if col not in ["CVE", "exploitable"]], axis=1, inplace=True)
    #print(exploits_df.columns)
    exploits_df.drop_duplicates(subset=["CVE"], inplace=True)
    exploits_df.set_index("CVE", inplace=True)
    exploits_df.index.name = "cve"
    exploits_df.rename({col: "EDB_" + col for col in exploits_df.columns}, axis=1, inplace=True)
   #print(exploits_df)

    #exit()

    # LEFT JOIN as if no exploit was found it is treated as NOT_EXPLOITED in EDB
    dependent_df = nvd_df.join(exploits_df, how="left")
    dependent_df["EDB_exploitable"] = dependent_df["EDB_exploitable"].fillna(False)

    #print(dependent_df)
    #exit()

    return load_features(dependent_df, dir)

''' 
Choose the exact dataset to be loaded
'''
'''def scompatta_dataset(choosen):
    if choosen == "bow_dir":
        bow_dir = "data/features/bow/"
        if not os.path.exists(os.path.join(bow_dir, "data.npz")):
            print("Missing file with BoW values")
            exit(1)
        bow_df = prepare_dataset_for_ml(bow_dir)
        return bow_df
    
    elif choosen == "tf_dir":
        tf_dir = "data/features/tf/"
        if not os.path.exists(os.path.join(tf_dir, "data.npz")):
            print("Missing file with TF values")
            exit(1)
            tf_df = prepare_dataset_for_ml(tf_dir)
        return tf_df

    elif choosen == "tfidf_dir":
        tfidf_dir = "data/features/tfidf/"
        if not os.path.exists(os.path.join(tfidf_dir, "data.npz")):
            print("Missing file with TF-IDF values")
            exit(1)
        tfidf_df = prepare_dataset_for_ml(tfidf_dir)
        return tfidf_df
'''

''' 
Transform dependent variables for (0,1) -> (-1,1)
'''
def transform_dependent_values(y_data):
    y_data = y_data.values.ravel()
    y_data = np.where(y_data, 1, -1)

    return y_data


''' 
Main program section 
'''
if __name__ == "__main__":
    bow_dir = "data/features/bow/"
    tf_dir = "data/features/tf/"
    tfidf_dir = "data/features/tfidf/"
    if not os.path.exists(os.path.join(bow_dir, "data.npz")):
        print("Missing file with BoW values")
        exit(1)
    if not os.path.exists(os.path.join(tf_dir, "data.npz")):
        print("Missing file with TF values")
        exit(1)
    if not os.path.exists(os.path.join(tfidf_dir, "data.npz")):
        print("Missing file with TF-IDF values")
        exit(1)
    
    #######################################################
    # We are working here only with tfidf_df for now      #
    #######################################################

    #bow_df = prepare_dataset_for_ml(bow_dir)
    #tf_df = prepare_dataset_for_ml(tf_dir)
    tfidf_df = prepare_dataset_for_ml(tfidf_dir)

    '''print('Cols :', bow_df.columns)
    print('bow_df :\n', bow_df)

    print('Cols :', tf_df.columns)
    print('tf_df :\n', tf_df)

    print('Cols :', tfidf_df.columns)
    print('tfidf_df :\n', tfidf_df)

    print('Inizio la creazione dei csv')
    bow_df.to_csv(r'bow_df.csv', index=False, header=True)
    print('finito bow_df')
    tf_df.to_csv(r'tf_df.csv', index=False, header=True)
    print('finito td_df_df')
    tfidf_df.to_csv(r'tfidf_df.csv', index=False, header=True)
    print('finito tfidf_df_df')'''
    
    print('*** END ***')


''' 
Assign test train percentage & dataset name
'''
# percentage of dataset for TEST,0
percentage = 0.20
dataset="tfidf_df"

#tfidf_df.shape[0]

#tfidf_df["EDB_exploitable"]

choosen=tfidf_df

#######################################
# Removing label from dataset columns #
#######################################

print('Creating the list of columns without the DEPENDENT variable... ')
cols = list(choosen.columns)
cols.remove('EDB_exploitable')
choosen['EDB_exploitable'] = choosen['EDB_exploitable'].astype(int)

###############################################
# Split columns for Training and Test dataset #
###############################################
print('Splitting dataset into Training and Testing sets')
x_train,x_test,y_train,y_test=train_test_split(choosen[cols],choosen[['EDB_exploitable']],test_size=percentage,shuffle=True)
print('Done splitting')

###############################################
# Convert DEPENDENT variable into values 1/-1 #
###############################################

y_train = transform_dependent_values(y_train)
y_test = transform_dependent_values(y_test)


################################################
# Create svc model and fit it on training data #
################################################

##################################################
# SVC gave problems - Sparse Matrix incompatible #
##################################################

'''print('training svc model... ')
svc = SVC(kernel='linear', C = 1.0)
start = timer()
svc.fit(x_train, y_train)
end = timer()
print(f'SVC training time in seconds: {(end - start)}')
tempo = (end - start)'''

'''# sklearn.svm.LinearSVC

#X, y = make_classification(n_features=2515, random_state=0)
#clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5))
#clf.fit(x_train, y_train) '''


'''start = time.time()
# Data Training
# Normal / SVC
print("Training Normal/SVC ...")
clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
clf.fit(x_train, y_train)
end = time.time()

tempo_normal = end - start
print(tempo_normal)'''

'''
#####################################################
# Create Adaboost model and fit it on training data #
#####################################################
#-------------TRAIN CON ADABOOST DECOMMENTARE E COMMENTARE SVC
nest=50
start = timer()
clf_AB = AdaBoostClassifier(base_estimator=None, n_estimators=nest, algorithm='SAMME.R', random_state=None)
clf_AB.fit(x_train, y_train)
end = timer()
tempo_normal = end - start
print(tempo_normal)

##########################################################
# Create Random Forest model and fit it on training data #
##########################################################
#-------------TRAIN With Random Forest  ------------------------
leaf_depth=20
start = timer()
clf_RF = RandomForestClassifier(max_depth=leaf_depth, random_state=0)
clf_RF.fit(x_train, y_train)
end = timer()
tempo_normal = end - start
print(tempo_normal)


# ADABOOST
####################################
# Predict accuracy on training set #
####################################
print('Predicting on training set - AdaBoost')
predictions = clf_AB.predict(x_train)
print("Reporting...")
report = classification_report(y_train, predictions)
with open('report/report_' + dataset + '.txt', 'a') as f:
    f.write('Adaboost Training Set \n')
    f.write(report)
    f.write('Training Time ('+str(nest)+'):'+str(tempo_normal)+'\n')
    f.write('\n')
f.close()
print("Saved...")

##################################
# Predict results on testing set #
##################################
print('Predicting on Test set - AdaBoost')
start = timer()
predictions = clf_AB.predict(x_test)
end = timer()
tempo = (end - start)
print("Reporting...")
report = classification_report(y_test, predictions)
with open('report/report_' + dataset + '.txt', 'a') as f:
    f.write('Adaboost Test Set \n')
    f.write(report)
    f.write('Test Set Prediction Time ('+str(nest)+'):'+str(tempo)+'\n')
    f.write('\n')
f.close()
print("Saved...")


# RANDOM FOREST
####################################
# Predict accuracy on training set #
####################################
print('Predicting on Training set - Random Forest')
predictions = clf_RF.predict(x_train)
print("Reporting...")
report = classification_report(y_train, predictions)
with open('report/report_' + dataset + '.txt', 'a') as f:
    f.write('Random Forest Training Set \n')
    f.write(report)
    f.write('Training Time ('+str(leaf_depth)+'):'+str(tempo_normal)+'\n')
    f.write('\n')
f.close()
print("Saved...")

##################################
# Predict results on testing set #
##################################
print('Predicting on Test set - Random Forest')
start = timer()
predictions = clf_RF.predict(x_test)
end = timer()
tempo = (end - start)
print("Reporting...")
report = classification_report(y_test, predictions)
with open('report/report_' + dataset + '.txt', 'a') as f:
    f.write('Random Forest Test Set \n')
    f.write(report)
    f.write('Test Set Prediction Time ('+str(leaf_depth)+'):'+str(tempo)+'\n')
    f.write('\n')
f.close()
print("Saved...")'''


##########################
# Define Quantum sampler #
##########################

#dwave_sampler = DWaveSampler(token=DEVtoken)
#emb_sampler = EmbeddingComposite(dwave_sampler(profile="europe"))
#lmd = 0.04

#####################
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
    f.write('QBoost Training Set \n')
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
    f.write('QBoost Test Set \n')
    f.write(report)
    f.write('tempo qboost testing: '+str(tempo)+'\n')
f.close()
print("saved...")

