  _____  ______          _____  __  __ ______  
 |  __ \|  ____|   /\   |  __ \|  \/  |  ____| 
 | |__) | |__     /  \  | |  | | \  / | |__    
 |  _  /|  __|   / /\ \ | |  | | |\/| |  __|   
 | | \ \| |____ / ____ \| |__| | |  | | |____  
 |_|  \_\______/_/    \_\_____/|_|  |_|______| 
                                               
                                              
Utilizzo: 
1. Legere main.py, all'inizio del file ci sono i parametri da compilare con il DEVtoken di DwaveLeap e la percentuale di utilizzo del dataset (se si vuole utilizzare partizionato).
2. è necessaria una cartella "report", ad ogni run verrà creato un file "report.txt" all'interno della cartella con matrice di confusione e tempo impiegato
3. Decommentare a scelta uno tra i 3 dataset forniti: 
#dataset = "bow_dir"
#dataset = "tf_dir"
#dataset = "tfidf_dir"

Risultati: 
Vengono riportate le matrici di confusione di TEST su training set e di TEST su testing set.
Vengono riportati i tempi di Training (ATTENZIONE NON DEL TEST SUL TRAINING SET MA DI ADDESTRAMENTO) e di Testing.

File necessari:
1. Nella cartella data ci sono i requirements delle librerie per estrarre il dataset
2. i file necessari sono:
	2.1. Cartella 'data' per intero
	2.2. main.py
	2.3. cartella report
	2.4. prepare_dataset.py
	2.5. cartella qboost