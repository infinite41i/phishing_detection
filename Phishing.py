CLF_RUNS = 3 #!make this 10!

import pandas as pd
import numpy as np
from scipy.io import arff

# from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

initial_data = arff.loadarff('./Training_Dataset.arff')

phishing_data = pd.DataFrame(initial_data[0])
data = phishing_data.iloc[:, 0:30]
target = phishing_data.iloc[:, 30]
print(data.info())

#Random Forest Classifier
def run_rfc():
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    y = clf.predict(X_test)
    
    res = [precision_score(y_test, y, pos_label='1')*100,
               recall_score(y_test, y, pos_label='1')*100,
               f1_score(y_test, y, pos_label='1')*100, 
               accuracy_score(y_test, y)*100]
    return res

#Gaussian Naive Bayesian
def run_bayes():
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    y = clf.predict(X_test)
    
    res = [precision_score(y_test, y, pos_label='1')*100,
               recall_score(y_test, y, pos_label='1')*100,
               f1_score(y_test, y, pos_label='1')*100,
               accuracy_score(y_test, y)*100]
    return res

#Multi Layer Perceptron
def run_mlp():
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    clf = MLPClassifier(solver='lbfgs')
    clf.fit(X_train, y_train)
    
    y = clf.predict(X_test)
    
    res = [precision_score(y_test, y, pos_label='1')*100,
               recall_score(y_test, y, pos_label='1')*100,
               f1_score(y_test, y, pos_label='1')*100,
               accuracy_score(y_test, y)*100]
    return res

# Support Vector Machine
def run_svm():
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    
    y = clf.predict(X_test)
    
    res = [precision_score(y_test, y, pos_label='1')*100,
               recall_score(y_test, y, pos_label='1')*100,
               f1_score(y_test, y, pos_label='1')*100, 
               accuracy_score(y_test, y)*100]
    return res

def run_k_fold(func):
    partial_results = [func() for _ in range(CLF_RUNS)]
    res = np.average(partial_results, axis=0)
    return res

def run_experminet():
    res_dict = {'RFC': run_k_fold(run_rfc),
                'BYS': run_k_fold(run_bayes),
                'MLP': run_k_fold(run_mlp),
                'SVM': run_k_fold(run_svm)}

    rows = ['Precision', 'Recall', 'F-score', 'Accuracy'] #add time
    res = pd.DataFrame(res_dict, index=rows)
    return res

def main():
    res = run_experminet()
    print(res)

if(__name__ == "__main__"):
    main()