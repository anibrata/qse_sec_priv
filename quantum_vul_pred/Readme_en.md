# Dataset for Quantum Algorithms

## Exploitability Prediction

The dataset consists of 147,936 rows, each representing a known vulnerability described by means of a CVE. For each, textual data was collected (in unstructured natural language) and extracted a set of textual features, each representing a term (after having processed the text with the classic preparation pipeline, made up of tokenization, stemming and stop word removal ).

The total number of features is: 2515

In the work we have done we have built three identical datasets (with the same rows and the same number of features), but which differ only in the values ​​that the features assume:
- * Dataset BoW *, with integer values, representing the absolute count of the terms;
- * Dataset TF *, with real values, representing the relative count of the terms;
- * TF-IDF dataset *, with real values, representing the TF-IDF value of the terms.

The three datasets are NOT available as csvs due to their huge size, but stored as follows:
- `bow / data.npz`, Scipy sparse matrix with the features for the BoW dataset;
- `tf / data.npz`, Scipy sparse matrix with the features for the TF dataset;
- `tfidf / data.npz`, Scipy sparse matrix with the features for the TF-IDF dataset;
- `cves_exploits.csv`, CSV file with the values ​​of the dependent variable" EDB_exploitable "

To obtain the three datasets in the form of Pandas Dataframe it is necessary to execute the `prepare_datasets.py` script with the following command:` python prepare_datasets.py`.
After about ten seconds, the three datasets will be loaded into the variables `bow_df`,` tf_df`, `tfidf_df` and printed on the terminal.