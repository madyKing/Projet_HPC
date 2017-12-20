# coding : utf-8
import pandas as pd
import sys
from mpi4py import MPI
sys.path.insert(0, '/user/7/.base/yenden/home/Documents/HPC-Projects/Projet_HPC/TreeMethods')
#sys.path.insert(0, '/home/mamady/Bureau/Cours3A/HPC/Projet RF/codeBase/Projet_HPC/TreeMethods')

from DecisionTreeClassifier import DecisionTreeClassifier
from RandomForestClassifier import RandomForestClassifier
dataset = [[2.771244718, 1.784783929, 0],
		       [1.728571309, 1.169761413, 0],
		       [3.678319846, 2.81281357, 1],
		       [3.961043357, 2.61995032, 1],
		       [2.999208922, 2.209014212, 0],
		       [7.497545867, 3.162953546, 0],
		       [9.00220326, 3.339047188, 1],
		       [7.444542326, 0.476683375, 1],
		       [10.12493903, 3.234550982, 0],
		       [6.642287351, 3.319983761, 1]]

data_point = pd.Series([2.0, 23.0], index=['feature_1','feature_2'])
data_test = [[6.642287351, 3.319983761],[3.678319846, 2.81281357], [10.12493903, 3.234550982]]

df = pd.DataFrame(data=dataset,columns =['feature_1','feature_2','target'])

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

start = MPI.Wtime()
forest = RandomForestClassifier(n_trees=size,max_depth=5, min_size=1)
forest.fit(df, target='target')

if rank==0:
	predict = forest._predict(data_test)
	finish = MPI.Wtime()
	print("predictions", predict)
	print("\ntime ", finish-start)
