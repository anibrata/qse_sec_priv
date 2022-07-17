import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qboost import QBoostClassifier
#from dwave.system.samplers import DWaveSampler
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from dwave.system.composites import EmbeddingComposite
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier

n_uno = [55000, 55000]
n_zero = [200000, 10000]

n_train_zero = 5000
n_train_uno = 5000

# QBOOST classification paramenters
#DW_PARAMS = {'num_reads': 3000,
#             'auto_scale': True,
#             # "answer_mode": "histogram",
#             'num_spin_reversal_transforms': 10,
#             # 'annealing_time': 10,
#             # 'postprocess': 'optimization',
#             }
#NUM_WEAK_CLASSIFIERS = 35
#TREE_DEPTH = 3
#dwave_sampler = DWaveSampler(token="DEV-2df3cc6405aa02a28281d11d8112bae3a2597df5")
# sa_sampler = micro.dimod.SimulatedAnnealingSampler()
#emb_sampler = EmbeddingComposite(dwave_sampler)
#lmd = 0.5


# End Parameters


# loadData loads all CSV data to memory and substitutes the missing data with the maximum in its column
# this function processes data to replace 0 with -1 and keeps 1 as 1, also does what is mentioned above this line
# Finally loads data in the variable "data"
def loadData(path):
    data = pd.read_csv(path)
    print('number of rows: ',len(data))
    data['label'] = data['label'].replace(0, -1) # replaces 0 with -1
    data['label'] = data['label'].replace(1, 1) # replaces 1 with 1
    #print(data)
    return data


def independentListFind(data):
    print('*********** Printing details of data for the function')
    #print(data)
    shape = data.shape
    #print(shape)
    cols = list(data.columns.values)
    print(cols)
    independentList = cols[0:shape[1] - 1]
    print(independentList)
    cols.pop(-1)
    #print(cols)
    print('********** END Printing details of data for the function')
    return independentList, cols


# scaleData takes care of scaling all the data within the data in order to make it between 0 and 1
def scaleData(data, independentList):
    data['time'] = data['time'].str.replace(':', '')
    data['time'] = [int(x) for x in data['time']]
    scaler = MinMaxScaler()
    # Discretization of high and low values in Iot_Fridge.csv
    data['temp_condition'] = data['temp_condition'].str.replace('high', '1.00')
    data['temp_condition'] = data['temp_condition'].str.replace('low', '0.00')
    data['date'] = data['date'].str.replace("-", "")
    data['date'] = data['date'].str.replace("Gen", "01")
    data['date'] = data['date'].str.replace("Feb", "02")
    data['date'] = data['date'].str.replace("Mar", "03")
    data['date'] = data['date'].str.replace("Apr", "04")
    data['date'] = data['date'].str.replace("May", "05")
    data['date'] = data['date'].str.replace("Jun", "06")
    data['date'] = data['date'].str.replace("Jul", "07")
    data['date'] = data['date'].str.replace("Aug", "08")
    data['date'] = data['date'].str.replace("Sep", "09")
    data['date'] = data['date'].str.replace("Oct", "10")
    data['date'] = data['date'].str.replace("Nov", "11")
    data['date'] = data['date'].str.replace("Dic", "12")

    # print(data[independentList])
    data[independentList] = scaler.fit_transform(data[independentList])
    return data

##################### MAIN PROCESSING STARTS #####################

# SELECTION
# Load Dataset as CSV - PRE-ELABORATION

print('Start...')
simpath = 'IoT_Fridge.csv'
path = os.path.abspath(simpath)
data_csv = loadData(path) # LOAD DATA
label = 'label'

print('FILE PATH: ', simpath)
print('ABSOLUTE PATH: ',path)
#print(data_csv)

############################################# TRAINING ##################################################
print("TRAIN--------------------------------------------------------------------------------------TRAIN")

# selection of the sample number for the train (balanced with n_train_uno and n_train_zero)
data = data_csv
print("Data Selection...")

# Select NORMAL activities and store in data_train_zero
data_train_zero = data.loc[data['label'] == -1]
print(data_train_zero, len(data_train_zero))

# Randomly choosing "n_train_zero" rows for training NORMAL situations random 5000 numbers
#print(n_train_zero)
chosen = np.random.choice(n_train_zero, replace=True, size=n_train_zero)  # WHY replace = True ??
#print(chosen, len(chosen))
# Question: why are random numbers selected with REPETITION in 5000 range ?
# We can randomly select 5000 rows from NORMAL activities
# Also, we can change the train and test to be 70:30


# Print all random numbers selected
# for i in chosen: print(i,', ')

#print(data_train_zero)
data_train_zero = data_train_zero.iloc[chosen] # select all random normal rows and store in data_train_zero (5000 rows)
print('size of train NORMAL : ', len(data_train_zero))

# Select ABNORMAL activities and store in data_train_zero
data_train_uno = data.loc[data['label'] == 1]
#print(n_train_uno)
chosen = np.random.choice(n_train_uno, replace=True, size=n_train_uno)
#print(chosen, len(chosen))

data_train_uno = data_train_uno.iloc[chosen] # select all random attack rows and store in data_train_zero (5000 rows)
print('size of train ATTACK : ',len(data_train_uno))

# Appending the normal data and abnormal data together (Why now?)
# Why can't we try traditionally like taking the complete dataset edit and divide it into 80:20 or 70:30 ?
# Answering the earlier question: It is to keep the count of number of normal and attack rows for training and testing efficiency.
data_train = data_train_zero.append(data_train_uno, ignore_index=True)
#data_train = pd.concat(data_train_zero,data_train_uno, ignore_index=True)
print('Length of Training data: ')
print('size of ALL TRAIN : ',len(data_train))
print("End of data selection")
# End of selection of sample numbers


# Selection of the independent list and scaling the data (Data Transformation)
# Starting data processing and transformation
print("Start Data Transformation")
independentList, cols = independentListFind(data)
#print(independentList)
#print(cols)

## Why are we using this function ? its returning same values to both the variables
independentList_train, cols_train = independentListFind(data_train)
print('Independent List: ', independentList_train)
print('Training Columns: ', cols_train)


print('*********************** PRINTING DATA TRANSFORMATION ********************')
labels_train = data_train['label']
print(labels_train)
print("Data_train BEFORE scaling: ")
print(data_train[independentList_train], data_train['type'])
data_train = scaleData(data_train, independentList)
print("Data_train AFTER scaling: ")
print(data_train)
data_train = data_train[independentList_train]
print("Data_train only specific columns: ")
print(data_train)
data_train = data_train.to_numpy() # changes the data to pandas DF format
print("Data_train after NUMPY: ")
print(data_train)
data_train = data_train[:, :-1] # Remove the last column from the data_train DF
print("Data_train after [:, :-1] : ") # DF without label column
print(data_train)
print("End Data Transformation")
# End of Data Transformation

# Data Training
# Normal / SVC
print("Training Normal/SVC ...")
# ------------TRAIN CON SVC DECOMMENTARE E COMMENTARE ADABOOST
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(data_train, labels_train)
# --------------------------------------------------------------

# ------------TRAIN CON ADABOOST DECOMMENTARE E COMMENTARE SVC
#clf = AdaBoostClassifier(base_estimator=None, n_estimators=35, algorithm='SAMME.R', random_state=None)
#clf.fit(data_train, labels_train)
print("Completato")
# --------------------------------------------------------------

# Quantum
#print("Training Quantum...")
# qboost = QBoostClassifier(data_train, labels_train, 0.01)
#qboost = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
#qboost.fit(data_train, labels_train, emb_sampler, lmd=lmd, **DW_PARAMS)
#print("Completato")

# TEST----------------------------------------------------------------------------TEST--------------------------------------------------------------------------TEST
print(
    "TEST----------------------------------------------------------------------------TEST--------------------------------------------------------------------------TEST")
for i in range(0, len(n_uno)):
    data = data_csv
    print("data selection...")
    # selezione del numero di sample per il test (bilanciabile con n_uno ed n_zero)
    data_zero = data.loc[data['label'] == -1]
    chosen = np.random.choice(n_zero[i], replace=True, size=n_zero[i])
    data_zero = data_zero.iloc[chosen]
    data_uno = data.loc[data['label'] == 1]
    chosen = np.random.choice(n_uno[i], replace=True, size=n_uno[i])
    data_uno = data_uno.iloc[chosen]
    data = data_zero.append(data_uno, ignore_index=True)
    print("fine data selection")
    # fine selezione del numero di sample

    # selezione dell'indipendent list e scalare i dati (Data Transformation)
    print("data Transformation")
    independentList, cols = independentListFind(data)
    data_pandas = data[independentList]
    labels = data['label']
    data = scaleData(data, independentList)
    data = data[independentList]
    data = data.to_numpy()
    data = data[:, :-1]
    print("fine data Transformation")
    # fine Data Transformation

    # DATA MINING
    # --------------NORMAL---------------------------------

    print("Data Mining Normale in corso...")
    start = time.time()
    normal_predicted = clf.predict(data)
    print("Completato")
    end = time.time()
    print("Tempo impiegato: ")
    tempo_normal = end - start
    print(end - start)
    print(normal_predicted)

    # --------------QUANTUM-----------------------------------
    #print("Data mining Quantum in corso...")
    #start = time.time()
    #quantum_predicted = qboost.predict(data)
    #print("Completato")
    #end = time.time()
    #print("Tempo impiegato: ")
    #print(quantum_predicted)
    #tempo_quantum = end - start
    #print(end - start)

    for j in range(0, len(normal_predicted)):
        if normal_predicted[j] == -1:
            normal_predicted[j] = 0

#    for j in range(0, len(quantum_predicted)):
#        if quantum_predicted[j] < 0.3:
#            quantum_predicted[j] = 0
#        if quantum_predicted[j] >= 0.3:
#            quantum_predicted[j] = 1

    for j in range(0, len(labels)):
        if labels[j] == -1:
            labels[j] = 0

    # print(normal_predicted)
    # print(quantum_predicted)
    # print(labels)
    # FINE DATA MINING

    # DATA EVALUATION
    print("Plotting Normal...")
    from datetime import datetime

    label = labels
    data_normal = pd.DataFrame(data, columns=['date', 'time', 'fridge_temperature', 'temp_condition'])
    data_normal['time'] = data_pandas['time'].str.strip()
    data_normal['time'] = pd.to_datetime(data_normal['time'], format='%H:%M:%S').dt.hour
    data_normal['fridge_temperature'] = data_pandas['fridge_temperature']
    data_normal['label'] = normal_predicted
    data_rosso = data_normal[normal_predicted == 1]
    data_blu = data_normal[normal_predicted == 0]
    data_rosso_time = data_rosso['time']
    data_rosso_temperature = data_rosso['fridge_temperature']
    data_blu_time = data_blu['time']
    data_blu_temperature = data_blu['fridge_temperature']
    red = plt.scatter(data_rosso_time, data_rosso_temperature, color='red', alpha=0.5)
    blue = plt.scatter(data_blu_time, data_blu_temperature, color='blue', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Fridge Temperature')
    plt.legend((red, blue), ('attacco', 'normale'))
    plt.title(str(n_uno[i]) + ' attacchi ' + str(n_zero[i]) + ' normali - Classificazione normale', weight='bold')
    plt.savefig('plt/plt_' + str(n_uno[i]) + 'attacchi' + str(n_zero[i]) + 'normali_Normal.jpg')
    print("file saved")

    print("Plotting Quantum...")
    label = labels
    data_normal = pd.DataFrame(data, columns=['date', 'time', 'fridge_temperature', 'temp_condition'])
    data_normal['time'] = data_pandas['time'].str.strip()
    data_normal['time'] = pd.to_datetime(data_normal['time'], format='%H:%M:%S').dt.hour
    data_normal['fridge_temperature'] = data_pandas['fridge_temperature']
 #   data_normal['label'] = quantum_predicted
 #   data_rosso = data_normal[quantum_predicted == 1]
 #   data_blu = data_normal[quantum_predicted == 0]
    data_rosso_time = data_rosso['time']
    data_rosso_temperature = data_rosso['fridge_temperature']
    data_blu_time = data_blu['time']
    data_blu_temperature = data_blu['fridge_temperature']
    plt.scatter(data_rosso_time, data_rosso_temperature, color='red', alpha=0.5)
    plt.scatter(data_blu_time, data_blu_temperature, color='blue', alpha=0.5)
    plt.legend((red, blue), ('attacco', 'normale'))
    plt.title(str(n_uno[i]) + ' attacchi ' + str(n_zero[i]) + ' normali - Classificazione Quantum', weight='bold')
    plt.xlabel('Time')
    plt.ylabel('Fridge Temperature')
    plt.savefig('plt/plt_' + str(n_uno[i]) + 'attacchi_' + str(n_zero[i]) + 'normali_Quantum.jpg')
    print("file saved")

    print("Reporting Normal...")
    from sklearn.metrics import classification_report

    report = classification_report(labels, normal_predicted)
    with open('report/report_' + str(n_uno[i]) + 'attacchi_' + str(n_zero[i]) + 'normali.txt', 'a') as f:
        print(f.write(report))
        print(f.write('tempo normale: ' + str(tempo_normal) + '\n'))
    f.close()
    print("saved...")

    print("Reporting Quantum...")
  #  report = classification_report(labels, quantum_predicted)
    print(report)
    with open('report/report_' + str(n_uno[i]) + 'attacchi_' + str(n_zero[i]) + 'normali.txt', 'a') as f:
        print(f.write(report))
  #      print(f.write('tempo quantum: ' + str(tempo_quantum)))
    f.close()
    print("saved...")
    # FINE DATA EVALUATION
print('FINE')
