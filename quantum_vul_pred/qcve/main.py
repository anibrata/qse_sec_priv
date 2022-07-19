from classify.classificator import Classifier
from sklearn.model_selection import train_test_split
from prepare_datasets import scompatta_dataset


def convert_label_for_quantum(y_data):
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

if __name__ == '__main__':
    list_dataset = ["bow_dir", "tf_dir", "tfidf_dir"]

    for dataset in list_dataset:

        choosen = scompatta_dataset(dataset)
        print('Dataset loaded')

        choosen['EDB_exploitable'] = choosen['EDB_exploitable'].astype(int)
 
        choosen['EDB_exploitable'] = convert_label_for_quantum(choosen['EDB_exploitable'])

        print('splitting dataset... ')

        X = choosen.drop(['EDB_exploitable'], axis=1)
        y = choosen['EDB_exploitable']

        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y)

        classifier = Classifier(x_train, x_test, y_train, y_test)

        classifier.set_dataset(dataset)

        classifier.run_classification(False)