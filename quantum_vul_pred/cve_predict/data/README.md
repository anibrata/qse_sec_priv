# Dataset for Quantum Algorithms

## Exploitability Prediction

Il dataset si compone di 147,936 righe, ciascuna rappresentate una vulnerabilità nota descritta per mezzo di un CVE. Per ciascuna, sono stati raccolti dei dati testuali (in linguaggio naturale non strutturato) ed estratte un insieme di feature testuali, ciascuna rappresentante un termine (dopo aver elaborato il testo con la classica pipeline di preparazione, fatta di tokenizazzione, stemming e stop word removal).

Il numero totale di feature è: 2515

Nel lavoro che abbiamo svolto abbiamo costurito tre dataset identici (con le stesse righe e lo stesso numero di feature), ma che si differenziano soltanto per i valori che assumo le feature:
- *Dataset BoW*, con valori interi, rappresentanti il conteggio assoluto dei termini;
- *Dataset TF*, con valori reali, rappresentanti il conteggio relativo delle dei termini;
- *Dataset TF-IDF*, con valori reali, rappresentanti il valore TF-IDF dei dei termini.

I tre dataset NON sono disponibili come csv per via della loro taglia enorme, ma memorizzati come segue:
- `bow/data.npz`, matrice sparsa Scipy con le feature per il dataset BoW; 
- `tf/data.npz`, matrice sparsa Scipy con le feature per il dataset TF; 
- `tfidf/data.npz`, matrice sparsa Scipy con le feature per il dataset TF-IDF;
- `cves_exploits.csv`, file CSV con i valori della variabile dipendente "EDB_exploitable"

Per ottenere i tre dataset sottoforma di Pandas Dataframe è necessario eseguire lo script `prepare_datasets.py` con il seguente comando: `python prepare_datasets.py`.
Dopo una decina secondi, i tre dataset saranno caricati nelle variabili `bow_df`, `tf_df`, `tfidf_df` e stampati a terminale.