import os
import pandas as pd
from scipy.sparse import load_npz


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


def load_features(left_df, dir):
    # Remember that this is made of SparseDType columns
    right_df = load_dataframe(dir)
    right_df.index.name = "cve"
    print(f"Joining Independent Variables")
    full_df = left_df.join(right_df, how="left")
    return full_df


def prepare_dataset_for_ml(dir):
    nvd_filepath = "data/cves/cves.csv"
    exploits_filepath = "data/exploits/cves_exploits.csv"

    nvd_df = pd.read_csv(nvd_filepath, sep=";")
    nvd_df.drop(["pub_date", "description", "references", "cwe"], axis=1, inplace=True)
    nvd_df.set_index("cve", inplace=True)
    nvd_df.rename({col: "NVD_" + col for col in nvd_df.columns}, axis=1, inplace=True)

    exploits_df = pd.read_csv(exploits_filepath)
    exploits_df.drop([col for col in exploits_df.columns.tolist() if col not in ["CVE", "exploitable"]], axis=1, inplace=True)
    exploits_df.drop_duplicates(subset=["CVE"], inplace=True)
    exploits_df.set_index("CVE", inplace=True)
    exploits_df.index.name = "cve"
    exploits_df.rename({col: "EDB_" + col for col in exploits_df.columns}, axis=1, inplace=True)
    
    # LEFT JOIN as if no exploit was found it is treated as NOT_EXPLOITED in EDB
    dependent_df = nvd_df.join(exploits_df, how="left")
    dependent_df["EDB_exploitable"] = dependent_df["EDB_exploitable"].fillna(False)
    return load_features(dependent_df, dir)


def scompatta_dataset(choosen):
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
            #tf_df = prepare_dataset_for_ml(tf_dir)
        tf_df = prepare_dataset_for_ml(tf_dir)
        return tf_df

    elif choosen == "tfidf_dir":
        tfidf_dir = "data/features/tfidf/"
        if not os.path.exists(os.path.join(tfidf_dir, "data.npz")):
            print("Missing file with TF-IDF values")
            exit(1)
        tfidf_df = prepare_dataset_for_ml(tfidf_dir)
        return tfidf_df    


################################

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
    
    bow_df = prepare_dataset_for_ml(bow_dir)
    tf_df = prepare_dataset_for_ml(tf_dir)
    tfidf_df = prepare_dataset_for_ml(tfidf_dir)

'''   print(bow_df)
    print(tf_df)
    print(tfidf_df)
    print('Inizio la creazione dei csv')
    bow_df.to_csv(r'bow_df.csv', index=False, header=True)
    print('finito bow_df')
    tf_df.to_csv(r'tf_df.csv', index=False, header=True)
    print('finito td_df_df')
    tfidf_df.to_csv(r'tfidf_df.csv', index=False, header=True)
    print('finito tfidf_df_df')
    print('FINITO')'''

