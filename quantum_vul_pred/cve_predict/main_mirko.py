#############################
# import necessary packages #
#############################
import pandas as pd
import os
import numpy as np
from pyparsing import col
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from qboost import QBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from timeit import default_timer as timer
from collections import Counter
from sklearn.metrics import confusion_matrix
from prepare_datasets import scompatta_dataset
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


#######################
# Settings parameters #
#######################
DEVtoken = ''
percentage = 0.20  # percentage of dataset

##################
# Choose dataset #
##################
'''
Uncomment ONE of these datasets to use it
'''
# dataset = "bow_dir"
# dataset = "tf_dir"
# dataset = "tfidf_dir"
list_dataset = ["tfidf_dir"]
#######################
# Functions #
#######################


def conv_label(y_data):
    """
    :param y_data: labels 
    :return: array of numerical labels (1:non-malicious,-1:malicious)
    """
    result = []
    for values in y_data:
        if values == 0:
            result.append(1)
        else:
            result.append(-1)
    return result


################
# Load dataset #
################
for dataset in list_dataset:
    choosen = scompatta_dataset(dataset)
    print('Dataset loaded')

    #######################################
    # Removing label from dataset columns #
    #######################################
    choosen['EDB_exploitable'] = choosen['EDB_exploitable'].astype(int)

    result_label = conv_label(choosen['EDB_exploitable'])
    choosen['EDB_exploitable'] = result_label
    del result_label

    print('splitting dataset... ')

    X = choosen.drop(['EDB_exploitable'], axis=1)
    y = choosen['EDB_exploitable']

    #####################################
    # Split columns for independentList #
    #####################################

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y)

    ##########################################################
    # Create Random Forest model and fit it on training data #
    ##########################################################
    start = timer()
    rfc = RandomForestClassifier(max_depth=10, random_state=0)
    rfc.fit(x_train, y_train)
    end = timer()
    print(f'Random Forest training time in seconds: {(end - start)}')
    tempo = (end - start)

    ####################################
    # Predict accuracy on training set #
    ####################################
    print('predicting on training set svc...')
    predictions = rfc.predict(x_train)
    print("Reporting...")
    report = classification_report(y_train, predictions)
    with open('report/report_random_forest_' + dataset + '.txt', 'a') as f:
        f.write(report)
        f.write('tempo normale training: ' + str(tempo) + '\n')
    f.close()
    print("saved...")

    ##################################
    # Predict results on testing set #
    ##################################
    print('predicting Random Forest... ')
    start = timer()
    predictions = rfc.predict(x_test)
    end = timer()
    tempo = (end - start)
    print("Reporting...")
    report = classification_report(y_test, predictions)
    with open('report/report_random_forest_' + dataset + '.txt', 'a') as f:
        f.write(report)
        f.write('tempo normale testing: ' + str(tempo) + '\n')
    f.close()
    print("saved...")

    ##########################################
    # Create SVC and fit it on training data #
    ##########################################
    print("Training Normal/SVC ...")
    start = timer()
    clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
    clf.fit(x_train, y_train)
    end = timer()
    print(f'SVC training time in seconds: {(end - start)}')
    tempo = (end - start)

    ####################################
    # Predict accuracy on training set #
    ####################################
    print('predicting on training set svc...')
    predictions = clf.predict(x_train)
    print("Reporting...")
    report = classification_report(y_train, predictions)
    with open('report/report_svc_' + dataset + '.txt', 'a') as f:
        f.write(report)
        f.write('tempo normale training: ' + str(tempo) + '\n')
    f.close()
    print("saved...")

    ##################################
    # Predict results on testing set #
    ##################################
    print('predicting svc... ')
    start = timer()
    predictions = clf.predict(x_test)
    end = timer()
    tempo = (end - start)
    print("Reporting...")
    report = classification_report(y_test, predictions)
    with open('report/report_svc_' + dataset + '.txt', 'a') as f:
        f.write(report)
        f.write('tempo normale testing: ' + str(tempo) + '\n')
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
    # parametri di classificazione di qboost
    DW_PARAMS = {'num_reads': 2,
                'auto_scale': True,
                # "answer_mode": "histogram",
                'num_spin_reversal_transforms': 2,
                # 'annealing_time': 10,
                # 'postprocess': 'optimization',
                }
    NUM_WEAK_CLASSIFIERS = 2
    TREE_DEPTH = 1
    dwave_sampler = DWaveSampler(token=DEVtoken)
    # sa_sampler = micro.dimod.SimulatedAnnealingSampler()
    emb_sampler = EmbeddingComposite(dwave_sampler)
    lmd = 0.5
    # fine paramentri
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
    with open('report/report_qboost_' + dataset + '.txt', 'a') as f:
        f.write(report)
        f.write('tempo qboost training: ' + str(tempo) + '\n')
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
    with open('report/report_qboost_' + dataset + '.txt', 'a') as f:
        f.write(report)
        f.write('tempo qboost testing: ' + str(tempo) + '\n')
    f.close()
    print("saved...")

print("[+] The process is finished!")
