import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qboost import QBoostClassifier
from dwave.system.samplers import DWaveSampler
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dwave.system.composites import EmbeddingComposite
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
#parametri di classificazione di qboost
DW_PARAMS = {'num_reads': 3000,
                 'auto_scale': True,
                 # "answer_mode": "histogram",
                 'num_spin_reversal_transforms': 10,
                 # 'annealing_time': 10,
                 #'postprocess': 'optimization',
                 }
NUM_WEAK_CLASSIFIERS = 35
TREE_DEPTH = 3
dwave_sampler = DWaveSampler(token="DEV-2df3cc6405aa02a28281d11d8112bae3a2597df5")
# sa_sampler = micro.dimod.SimulatedAnnealingSampler()
emb_sampler = EmbeddingComposite(dwave_sampler)
lmd = 0.5
#fine paramentri

n_ddos = [1000,2000,3000,4000,5000,6000]
n_injection = [1000,2000,3000,4000,5000,6000]
n_password = [1000,2000,3000,4000,5000,6000]
n_backdoor = [1000,2000,3000,4000,5000,6000]
n_normal = [1000,2000,3000,4000,5000,6000]

n_ddos_train = 100
n_injection_train = 100
n_password_train = 100
n_backdoor_train = 100
n_normal_train = 100


# loadData si occupa di caricare in memoria tutti i dati all'interno del file csv
# e di sostituire i dati mancanti con il massimo nella sua colonna
def loadData(path):
    data = pandas.read_csv(path, na_values=['Infinity'], low_memory=False)
    data['type'] = data['type'].str.replace("ddos", "4")
    data['type'] = data['type'].str.replace("injection", "3")
    data['type'] = data['type'].str.replace("password", "2")
    data['type'] = data['type'].str.replace("backdoor", "1")
    data['type'] = data['type'].str.replace("normal", "0")
    return data


def independentListFind(data):
    shape = data.shape
    cols = list(data.columns.values)
    independentList = cols[0:shape[1]]
    #independentList.pop(0)
    #independentList.pop(0) ##cancello i primi due argomenti del dataset ADHOC per il mio dataset fridge_IoT
    #cols.pop(0)
    #cols.pop(0)
    cols.pop(-2)
    return independentList, cols


# scaleData si occupa di scalare tutti i dati all'interno di data in maniera da renderli compresi tra 0 e 1
def scaleData(data, independentList):
    scaler = MinMaxScaler()
    #discretizazione di high e low in Iot_Fridge.csv 
    data['temp_condition'] = data['temp_condition'].str.replace('high','1.00')
    data['temp_condition'] = data['temp_condition'].str.replace('low','0.00')
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
    
    data['time'] = data['time'].str.replace(":", "")
    
   
    #print(data[independentList])
    data[independentList] = scaler.fit_transform(data[independentList])
    return data

    
def pcaFind(data, n):
    pca = PCA(n_components = n)
    pca.fit(data)

    return pca


def applyPCA(pca, data, label, n):

    colonne = []
    data = pca.transform(data)
    for i in range(0, n):
        colonne.append('principal component ' + str(i + 1))

    principalDf = pandas.DataFrame(data=data, columns=colonne)
    data = pandas.concat([principalDf, label], axis=1)

    return data

def select_label (data, n_ddos, n_injection, n_password, n_backdoor, n_normal):
    
    data_ddos = data.loc[data['type'] == "4"]
    chosen = np.random.choice(n_ddos, replace = True, size = n_ddos)
    data_ddos = data_ddos.iloc[chosen]
    
    data_injection = data.loc[data['type'] == "3"]
    chosen = np.random.choice(n_injection, replace = True, size = n_injection)
    data_injection = data_injection.iloc[chosen]
    
    data_password = data.loc[data['type'] == "2"]
    chosen = np.random.choice(n_password, replace = True, size = n_password)
    data_password = data_password.iloc[chosen]
      
    data_backdoor = data.loc[data['type'] == "1"]
    chosen = np.random.choice(n_backdoor, replace = True, size = n_backdoor)
    data_backdoor = data_backdoor.iloc[chosen]
    
    data_normal = data.loc[data['type'] == "0"]
    chosen = np.random.choice(n_normal, replace = True, size = n_normal)
    data_normal = data_normal.iloc[chosen]
    
    data = data_ddos.append(data_injection, ignore_index=True)
    data = data.append(data_password, ignore_index=True)
    data = data.append(data_backdoor, ignore_index=True)
    data = data.append(data_normal, ignore_index=True)
    
    return data


def replacetypevalue (labels):

    labels_zero = labels
    labels_zero = labels_zero.replace("0","-1")
    labels_zero = labels_zero.replace("1","-1")
    labels_zero = labels_zero.replace("2","-1")
    labels_zero = labels_zero.replace("3","-1")
    labels_zero = labels_zero.replace("4","-1")
    #print(labels_zero)
    labels_uno = labels 
    labels_uno = labels_uno.replace("0", "-1")
    labels_uno = labels_uno.replace("2", "-1")
    labels_uno = labels_uno.replace("3", "-1")
    labels_uno = labels_uno.replace("4", "-1")
    #print(labels_uno)
    labels_due = labels
    labels_due = labels_due.replace("0", "-1")
    labels_due = labels_due.replace("1", "-1")
    labels_due = labels_due.replace("3", "-1")
    labels_due = labels_due.replace("4", "-1")
    labels_due = labels_due.replace("2", "1") 
    #print(labels_due)
    labels_tre = labels
    labels_tre = labels_tre.replace("0", "-1")
    labels_tre = labels_tre.replace("1", "-1")
    labels_tre = labels_tre.replace("2", "-1")
    labels_tre = labels_tre.replace("4", "-1")
    labels_tre = labels_tre.replace("3", "1")
    #print(labels_tre)
    labels_quattro = labels
    labels_quattro = labels_quattro.replace("0", "-1") 
    labels_quattro = labels_quattro.replace("1", "-1") 
    labels_quattro = labels_quattro.replace("2", "-1") 
    labels_quattro = labels_quattro.replace("3", "-1") 
    labels_quattro = labels_quattro.replace("4", "1") 

    return labels_zero, labels_uno, labels_due, labels_tre, labels_quattro


def replacetypevalue_quantum (labels):

    labels_zero = labels
    labels_zero = labels_zero.replace("0",-1)
    labels_zero = labels_zero.replace("1",-1)
    labels_zero = labels_zero.replace("2",-1)
    labels_zero = labels_zero.replace("3",-1)
    labels_zero = labels_zero.replace("4",-1)
    #print(labels_zero)

    labels_uno = labels 
    labels_uno = labels_uno.replace("0", -1)
    labels_uno = labels_uno.replace("2", -1)
    labels_uno = labels_uno.replace("3", -1)
    labels_uno = labels_uno.replace("4", -1)
    labels_uno = labels_uno.replace("1", 1)
    #print(labels_uno)
    labels_due = labels
    labels_due = labels_due.replace("0", -1)
    labels_due = labels_due.replace("1", -1)
    labels_due = labels_due.replace("3", -1)
    labels_due = labels_due.replace("4", -1)
    labels_due = labels_due.replace("2", 1) 
    #print(labels_due)
    labels_tre = labels
    labels_tre = labels_tre.replace("0", -1)
    labels_tre = labels_tre.replace("1", -1)
    labels_tre = labels_tre.replace("2", -1)
    labels_tre = labels_tre.replace("4", -1)
    labels_tre = labels_tre.replace("3", 1)
    #print(labels_tre)
    labels_quattro = labels
    labels_quattro = labels_quattro.replace("0", -1) 
    labels_quattro = labels_quattro.replace("1", -1) 
    labels_quattro = labels_quattro.replace("2", -1) 
    labels_quattro = labels_quattro.replace("3", -1) 
    labels_quattro = labels_quattro.replace("4", 1) 
    #print(labels_quattro)
    return labels_zero, labels_uno, labels_due, labels_tre, labels_quattro


#SELECTION
#caricamente del database come csv PRE-ELABORATION
print('caricamento file csv... ')
path = 'IoT_Fridge.csv'
data_csv = loadData(path)
print('completato')

#-------------------------------------------------------------------------------------------------------------------TRAIN--------------------------------------------------------------------------------------------------------------
print('#-------------------------------------------------------------------------------------------------------------------TRAIN--------------------------------------------------------------------------------------------------------------')
data = data_csv
print('Data Selection...')
print('selezione del numero di sample...')
data_train = select_label(data, n_ddos_train, n_injection_train, n_password_train, n_backdoor_train, n_normal_train)
print('completato')

#selezione dell'indipendent list e scalare i dati (Data Transformation)
print("data Transformation...")
independentList_train, cols_train = independentListFind(data_train)
labels_train = data_train['type']
data_train = scaleData(data_train, independentList_train)
data_train = data_train[independentList_train]
label = 'type'
print('completato')
#fine Data Transformation

#eliminando la colonna 'label' poichè i label sono 'type'
print('eliminazione label non-type... ')
data_train = data_train.to_numpy()
data_train = data_train[:,:-2]
print('completato')
#fine eliminazione

#sistemazione etichette dei type
print('sistemazione delle etichette type...')
labels_zero_train, labels_uno_train, labels_due_train, labels_tre_train, labels_quattro_train = replacetypevalue(labels_train)
labels_zero_train_quantum, labels_uno_train_quantum, labels_due_train_quantum, labels_tre_train_quantum, labels_quattro_train_quantum = replacetypevalue_quantum(labels_train)
print('completato')
#fine sistemazione

#Data Training
#Normal
print('Data Train Normale...')

clf_uno = AdaBoostClassifier(base_estimator=None, n_estimators=35, algorithm='SAMME.R', random_state=None)
clf_uno.fit(data_train, labels_uno_train)
clf_due = AdaBoostClassifier(base_estimator=None, n_estimators=35, algorithm='SAMME.R', random_state=None)
clf_due.fit(data_train, labels_due_train)
clf_tre = AdaBoostClassifier(base_estimator=None, n_estimators=35, algorithm='SAMME.R', random_state=None)
clf_tre.fit(data_train, labels_tre_train)
clf_quattro =AdaBoostClassifier(base_estimator=None, n_estimators=35, algorithm='SAMME.R', random_state=None)
clf_quattro.fit(data_train, labels_quattro_train)
print('completato')

#Quantum
print('Data Train Quantum...')
qboost_uno = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
qboost_uno.fit(data_train, labels_uno_train_quantum, emb_sampler, lmd=lmd, **DW_PARAMS)
qboost_due = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
qboost_due.fit(data_train,  labels_due_train_quantum, emb_sampler, lmd=lmd, **DW_PARAMS)
qboost_tre = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
qboost_tre.fit(data_train,  labels_tre_train_quantum, emb_sampler, lmd=lmd, **DW_PARAMS)
qboost_quattro = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
qboost_quattro.fit(data_train,  labels_quattro_train_quantum, emb_sampler, lmd=lmd, **DW_PARAMS)
print("Completato")


#-------------------------------------------------------------------------------------------------------------------TEST--------------------------------------------------------------------------------------------------------------
print('#-------------------------------------------------------------------------------------------------------------------TEST--------------------------------------------------------------------------------------------------------------')

for i in range(0, len(n_ddos)):
    data = data_csv
    print('Data Selection...')
    print('selezione del numero di sample...')
    data = select_label(data, n_ddos[i], n_injection[i], n_password[i], n_backdoor[i], n_normal[i])
    print('completato')

    #selezione dell'indipendent list e scalare i dati (Data Transformation)
    print("data Transformation...")
    independentList, cols = independentListFind(data)
    data_pandas = data[independentList]
    labels = data['type']
    data = scaleData(data, independentList)
    data = data[independentList]
    label = 'type'
    print('completato')
    #fine Data Transformation

    #eliminando la colonna 'label' poichè i label sono 'type'
    print('eliminazione label non-type... ')
    data = data.to_numpy()
    data = data[:,:-2]
    print('completato')
    #fine eliminazione

    #sistemazione etichette dei type
    print('sistemazione delle etichette type...')
    labels_zero, labels_uno, labels_due, labels_tre, labels_quattro = replacetypevalue(labels)
    labels_zero_quantum, labels_uno_quantum, labels_due_quantum, labels_tre_quantum, labels_quattro_quantum = replacetypevalue_quantum(labels)
    print('completato')
    #fine sistemazione


    #DATA MINING
    normal_predicted = labels_zero
    quantum_predicted = labels_zero_quantum
    #--------------NORMAL---------------------------------
    start = time.time()
    print("Data Mining Normale in corso...")
    normal_predicted_uno = clf_uno.predict(data)
    normal_predicted_due = clf_due.predict(data)
    normal_predicted_tre = clf_tre.predict(data)
    normal_predicted_quattro = clf_quattro.predict(data)
    print("Completato")
    print("Tempo impiegato: ")
    end = time.time()
    tempo_normal = end - start
    print(end - start)

    #discretizzazione dei dati Normali
    for j in range(0, len(normal_predicted_uno)):
        if normal_predicted_uno[j] == '1':
            normal_predicted[j] = '1'
        elif normal_predicted_due[j] == '1':
            normal_predicted[j] = '2'
        elif normal_predicted_tre[j] == '1':
            normal_predicted[j] = '3'
        elif normal_predicted_quattro[j] == '1':
            normal_predicted[j] = '4'
        elif labels_zero[j] == '-1':
            normal_predicted[j] = '0'

    #--------------QUANTUM-----------------------------------
    print("Data Mining Quantum in corso...")
    start = time.time()
    quantum_predicted_uno = qboost_uno.predict(data)
    end = time.time()
    tempo_quantum1 = end - start
    start = time.time()
    quantum_predicted_due = qboost_due.predict(data)
    end = time.time()
    tempo_quantum2 = end - start
    start = time.time()
    quantum_predicted_tre = qboost_tre.predict(data)
    end = time.time()
    tempo_quantum3 = end - start
    start = time.time()
    quantum_predicted_quattro = qboost_quattro.predict(data)
    end = time.time()
    tempo_quantum4 = end - start
    print("Completato")
    print("Tempo impiegato: ")
    #print(quantum_predicted)
    tempo_quantum = tempo_quantum1+ tempo_quantum2 + tempo_quantum3 + tempo_quantum4

    #discretizzazione dati quantum
    for k in range(0, len(quantum_predicted_uno)):
        if quantum_predicted_uno[k] >= 0.3:
            quantum_predicted[k] = '1'
        elif quantum_predicted_due[k] >= 0.3:
            quantum_predicted[k] = '2'
        elif quantum_predicted_tre[k] >= 0.3:
            quantum_predicted[k] = '3'
        elif quantum_predicted_quattro[k]  >= 0.3:
            quantum_predicted[k] = '4'
        elif labels_zero_quantum[k]  < 0.3:
            quantum_predicted[k] = '0'

    
        
    #FINE DATA MINING

    #DATA EVALUATION 
    #Normal
    print("-------------------------NORMAL------------------------------")
    label = labels
    data_normal = pandas.DataFrame(data, columns = ['date','time','fridge_temperature', 'temp_condition'])
    data_normal['time'] = data_pandas['time'].str.strip()
    data_normal['time'] = pandas.to_datetime(data_normal['time'],format= '%H:%M:%S' ).dt.hour
    data_normal['fridge_temperature'] = data_pandas['fridge_temperature']
    data_normal['type'] = normal_predicted
    data_rosso = data_normal[normal_predicted == "1"]
    data_blu = data_normal[normal_predicted == "0"]
    data_verde = data_normal[normal_predicted == "2"]
    data_nero = data_normal[normal_predicted == "3"]
    data_viola = data_normal[normal_predicted == "4"]
    plt.scatter(data_rosso['time'] , data_rosso['fridge_temperature'] , color = 'red')
    plt.scatter(data_blu['time'] , data_blu['fridge_temperature'] , color = 'blue')
    plt.scatter(data_verde['time'] , data_verde['fridge_temperature'] , color = 'green')
    plt.scatter(data_nero['time'] , data_nero['fridge_temperature'] , color = 'black')
    plt.scatter(data_viola['time'] , data_viola['fridge_temperature'] , color = 'purple')
    data_rosso_time = data_rosso['time']
    data_rosso_temperature = data_rosso['fridge_temperature']
    data_blu_time = data_blu['time']
    data_blu_temperature = data_blu['fridge_temperature']
    data_verde_time = data_verde['time']
    data_verde_temperature = data_verde['fridge_temperature']
    data_nero_time = data_nero['time']
    data_nero_temperature = data_nero['fridge_temperature']
    data_viola_time = data_viola['time']
    data_viola_temperature = data_viola['fridge_temperature']
    red = plt.scatter(data_rosso_time , data_rosso_temperature, color='red' , alpha=0.5)
    blue = plt.scatter(data_blu_time , data_blu_temperature,  color='blue', alpha=0.5)
    green = plt.scatter(data_verde_time , data_verde_temperature, color='green' , alpha=0.5)
    black = plt.scatter(data_nero_time , data_nero_temperature, color='black' , alpha=0.5)
    purple = plt.scatter(data_viola_time , data_viola_temperature, color='purple' , alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Fridge Temperature')
    plt.legend((red, blue, green, black, purple), ('ddos', 'normale','injection', 'password','backdoor'))
    plt.title(str(n_normal[i])+'n '+str(n_ddos[i])+'d '+str(n_injection[i])+'i '+str(n_password[i])+'p '+str(n_backdoor[i])+'b '+'- Classificazione normale', weight='bold')
    plt.savefig('plt/plt_'+str(n_normal[i])+' normali'+str(n_ddos[i])+' ddos'+str(n_injection[i])+' injection'+str(n_password[i])+' password'+str(n_backdoor[i])+' backdoor'+'_Normal.jpg')
    print("file saved")

    #Quantum
    print("-------------------------Quantum------------------------------")
    label = labels
    data_normal = pandas.DataFrame(data, columns = ['date','time','fridge_temperature', 'temp_condition'])
    data_normal['time'] = data_pandas['time'].str.strip()
    data_normal['time'] = pandas.to_datetime(data_normal['time'],format= '%H:%M:%S' ).dt.hour
    data_normal['fridge_temperature'] = data_pandas['fridge_temperature']
    data_normal['type'] = quantum_predicted
    data_rosso = data_normal[quantum_predicted == "1"]
    data_blu = data_normal[quantum_predicted == "0"]
    data_verde = data_normal[quantum_predicted == "2"]
    data_nero = data_normal[quantum_predicted == "3"]
    data_viola = data_normal[quantum_predicted == "4"]
    plt.scatter(data_rosso['time'] , data_rosso['fridge_temperature'] , color = 'red')
    plt.scatter(data_blu['time'] , data_blu['fridge_temperature'] , color = 'blue')
    plt.scatter(data_verde['time'] , data_verde['fridge_temperature'] , color = 'green')
    plt.scatter(data_nero['time'] , data_nero['fridge_temperature'] , color = 'black')
    plt.scatter(data_viola['time'] , data_viola['fridge_temperature'] , color = 'purple')
    data_rosso_time = data_rosso['time']
    data_rosso_temperature = data_rosso['fridge_temperature']
    data_blu_time = data_blu['time']
    data_blu_temperature = data_blu['fridge_temperature']
    data_verde_time = data_verde['time']
    data_verde_temperature = data_verde['fridge_temperature']
    data_nero_time = data_nero['time']
    data_nero_temperature = data_nero['fridge_temperature']
    data_viola_time = data_viola['time']
    data_viola_temperature = data_viola['fridge_temperature']
    red = plt.scatter(data_rosso_time , data_rosso_temperature, color='red' , alpha=0.5)
    blue = plt.scatter(data_blu_time , data_blu_temperature,  color='blue', alpha=0.5)
    green = plt.scatter(data_verde_time , data_verde_temperature, color='green' , alpha=0.5)
    black = plt.scatter(data_nero_time , data_nero_temperature, color='black' , alpha=0.5)
    purple = plt.scatter(data_viola_time , data_viola_temperature, color='purple' , alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Fridge Temperature')
    plt.legend((red, blue, green, black, purple), ('ddos', 'normale','injection', 'password','backdoor'))
    plt.title(str(n_normal[i])+'n '+str(n_ddos[i])+'d '+str(n_injection[i])+'i '+str(n_password[i])+'p '+str(n_backdoor[i])+'b'+'- Classificazione quantum', weight='bold')
    plt.savefig('plt/plt_'+str(n_normal[i])+'normali'+str(n_ddos[i])+'ddos'+str(n_injection[i])+'injection'+str(n_password[i])+'password'+str(n_backdoor[i])+'backdoor'+'_Quantum.jpg')
    print("file saved")

    #REPORT
    print("Reporting Normal...")
    report = classification_report(labels, normal_predicted)
    with open('report/report_'+str(n_normal[i])+' normali'+str(n_ddos[i])+' ddos'+str(n_injection[i])+' injection'+str(n_password[i])+' password'+str(n_backdoor[i])+' backdoor'+'.txt', 'a') as f:
        print(f.write(report))
        print(f.write('tempo normale: '+str(tempo_normal)+'\n'))
    f.close()
    print("saved...")

    print("Reporting Quantum...")
    report = classification_report(labels, quantum_predicted)
    print(report)
    with open('report/report_'+str(n_normal[i])+' normali'+str(n_ddos[i])+' ddos'+str(n_injection[i])+' injection'+str(n_password[i])+' password'+str(n_backdoor[i])+' backdoor'+'.txt', 'a') as f:
        print(f.write(report))
        print(f.write('tempo quantum: '+str(tempo_quantum)))
    f.close()
    print("saved...")
    #FINE DATA EVALUATION
print('FINE')


    
    

    

    

    


    

    



