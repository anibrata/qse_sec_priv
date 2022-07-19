from configuration import *
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from qboost import QBoostClassifier


class Classifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Init variables

        params: 
            X_train: the training dataset
            X_test: the test dataset
            y_train: the label of the training dataset
            y_test: the label of the test dataset
        """
        self.classifier_list = classificators_list
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset = '' # for CVE dataset [bow, tf, tfidf]
        self.clf = None # the classificator instance

    
    def set_dataset(self, dataset):
        """
        Set the dataset already in use. This is for CVE classification
        param:
            dataset: the dataset currently in use [bow, tf, tfidf]
        """
        self.dataset = dataset
    

    def randomforest_classifier(self):
        """
        Create an instance of the random forest

        return: instance of RandomForestClassifier
        """
        from sklearn.ensemble import RandomForestClassifier

        rfc = RandomForestClassifier(max_depth=randomforest_max_depth)
        return rfc.fit(self.X_train, self.y_train)


    def adaboost_classifier(self):
        """
        Create an instance of the adaboost

        return: instance of AdaBoostClassifier
        """
        from sklearn.ensemble import AdaBoostClassifier
        # from configuration import randomforest_max_depth

        adabclassifier = AdaBoostClassifier()
        return adabclassifier.fit(self.X_train, self.y_train)

    

    def svc_classifier(self):
        """
        Create an instance of the SVC

        return: instance of SVC
        """
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))
        clf.fit(self.X_train, self.y_train)

        return clf
    

    def qboost_classifier(self):
        """
        Create a classifier for QBoost
        """
        dwave_sampler = DWaveSampler()
        emb_sampler = EmbeddingComposite(dwave_sampler)
        lmd = 0.04
        dwave_sampler = DWaveSampler(token=DEVtoken)
        emb_sampler = EmbeddingComposite(dwave_sampler)
        lmd = 0.5
        qboost = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
        return qboost.fit(self.X_train, self.y_train, emb_sampler, lmd=lmd, **DW_PARAMS)
        

    def metrics(self, y, prediction):
        """
        Calculate the metrics with specific y_dataset and prediction of the classifier
        that are in used.
        """
        from sklearn.metrics import classification_report

        return classification_report(y, prediction)


    def create_file_report(self, end_time, name_cls, report, time_performance):
        """
        Create report file
        """
        import os
        path = f'{os.getcwd()}/report/{name_cls}_{self.dataset}.txt'
        with open(path, 'a') as f:
            f.write(report)
            f.write(f"Time performance for {time_performance}: {end_time}")
            f.write("\n")
        f.close()


    def learner_classifier(self, classifier: str):
        """
        Call dinamically the function randomforest, adaboost, and svc.
        All the methods are in the form nameclassifier_classifier i.e randomforest_classifier.
        Calculate the time for the prediction and create a file text with the results for each classifier.
        """
        from sys import modules
        from timeit import default_timer as timer

        if classifier.lower() not in self.classifier_list:
            raise Exception('Please, define a correct classificator. randomforest, adaboost, svc')
        
        start = timer()
        self.clf = getattr(self, '%s_classifier' % classifier.lower())()
        end = timer()
        end_time = end - start

        return end_time
    

    def run_classification(self, not_training_dataset: bool):
        """
        Run a classification and prediction for specific classifier.
        The classifier must be specified into configuration files.
        params:
            not_training_dataset: if False runs the prediction on training and testing dataset
                                  if True runs the prediction only on testing dataset
        """
        from timeit import default_timer as timer

        for clf in self.classifier_list:
            print(f'{clf} classification...')
            end_training_time = self.learner_classifier(clf)
            print(f'End {clf} training in: {end_training_time}')

            if not_training_dataset:
                print(f'Start prediction of {clf} for the test dataset...')
                prediction_test = self.clf.predict(self.X_test)
                report = self.metrics(self.y_test, prediction_test)
                self.create_file_report(end_training_time, clf, report, 'test dataset')
                print(f'End prediction test dataset of {clf} and report created.')

            else:
                print(f'Start prediction of {clf} for the training dataset...')
                prediction_train = self.clf.predict(self.X_train)
                report = self.metrics(self.y_train, prediction_train)
                self.create_file_report(end_training_time, clf, report, 'train dataset')
                print(f'End prediction of train dataset of {clf} and report created.')

                print(f'Start prediction of {clf} for the test dataset...')
                start = timer()
                prediction_test = self.clf.predict(self.X_test)
                end = timer()
                end_testing_time = end - start
                report = self.metrics(self.y_test, prediction_test)
                self.create_file_report(end_testing_time, clf, report, 'test dataset')
                print(f'End prediction of train dataset of {clf} and report created.')




        

    
