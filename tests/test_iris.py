import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import operator
from functools import reduce

import sys
from mpi4py import MPI

sys.path.insert(0, '/user/7/.base/yenden/home/Documents/HPC-Projects/Projet_HPC/TreeMethods')
#sys.path.insert(0, '/home/mamady/Bureau/Cours3A/HPC/Projet RF/codeBase/Projet_HPC/TreeMethods')

from DecisionTreeClassifier import DecisionTreeClassifier
from RandomForestClassifier import RandomForestClassifier


iris = datasets.load_iris()
features_iris = iris.data
target_iris = iris.target.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(features_iris, target_iris, test_size=0.33)
data_set = np.concatenate((X_train,y_train),axis=1)

l= reduce(operator.concat, y_test.T)


df = pd.DataFrame(data=data_set,columns =['feature_1','feature_2','feature_3','feature_4','target'])

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

start = MPI.Wtime()
forest = RandomForestClassifier(n_trees=size,max_depth=5, min_size=1)
forest.fit(df, target='target')

if rank==0 :
    predict = forest._predict(X_test)
    finish = MPI.Wtime()
    acc = accuracy_score(y_test, predict)
    print("acuuracy", acc)#, accuracy)
    print("\ntime ", finish-start)
