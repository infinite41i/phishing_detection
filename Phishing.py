CLF_RUNS = 10 #!make this 10!

import pandas as pd
import numpy as np
import arff
from time import time

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#import dataset
with open('./Training_Dataset.arff', 'r') as dataset_file:
    initial_data = arff.load(dataset_file, encode_nominal=True)
    dataset_file.close()

#split data and attributes
data_arr = np.array(initial_data['data'])
col_arr = np.array(initial_data['attributes'], dtype=object)
col_arr_names = col_arr[:, 0]

#make a dataframe using data and attributes
phishing_data = pd.DataFrame(data=data_arr, columns=col_arr_names)

#split data and target
data = phishing_data.iloc[:, 0:30]
target = phishing_data.iloc[:, 30]

def get_scores(y_test, y):
    res = [precision_score(y_test, y)*100,
               recall_score(y_test, y)*100,
               f1_score(y_test, y)*100, 
               accuracy_score(y_test, y)*100]
    return res

#Random Forest Classifier
def run_rfc():
    start_time = time()
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    y = clf.predict(X_test)
    
    scores = get_scores(y_test, y)
    
    total_time = time()-start_time
    
    return scores + [total_time]

#Gaussian Naive Bayesian
def run_bayes():
    start_time = time()
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    y = clf.predict(X_test)
    
    scores = get_scores(y_test, y)
    
    total_time = time()-start_time
    
    return scores + [total_time]

#Multi Layer Perceptron
def run_mlp():
    start_time = time()
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    clf = MLPClassifier(solver='lbfgs', max_iter=600, tol=0.001)
    clf.fit(X_train, y_train)
    y = clf.predict(X_test)
    
    scores = get_scores(y_test, y)
    
    total_time = time()-start_time
    
    return scores + [total_time]

# Support Vector Machine
def run_svm():
    start_time = time()
    X_train, X_test, y_train, y_test = train_test_split(data, target)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    
    y = clf.predict(X_test)
    
    scores = get_scores(y_test, y)
    
    total_time = time()-start_time
    
    return scores + [total_time]

def run_k_fold(func):
    partial_results = np.array([func() for _ in range(CLF_RUNS)])
    scores = np.average(partial_results[:,0:-1], axis=0)
    times = np.sum(partial_results[:, -1])
    res = np.concatenate((scores, times), axis=None)
    return res

def run_experminet():
    res_dict = {'RFC': run_k_fold(run_rfc),
                'BYS': run_k_fold(run_bayes),
                'MLP': run_k_fold(run_mlp),
                'SVM': run_k_fold(run_svm)}

    rows = ['Avg. Precision', 'Avg. Recall', 'Avg. F-score', 'Avg. Accuracy', 'Total Time'] #add time
    res = pd.DataFrame(res_dict, index=rows)
    return res

def main():
    res = run_experminet()
    print(res)

if(__name__ == "__main__"):
    main()