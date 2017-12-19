# coding : utf-8
import pandas as pd
import sys
sys.path.insert(0, '/home/mamady/Bureau/Cours3A/HPC/Projet RF/codeBase/RandomForests/TreeMethods')

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
data_test = [[6.642287351, 3.319983761]]

df = pd.DataFrame(data=dataset,columns =['feature_1','feature_2','target'])
# # tree = DecisionTreeClassifier(max_depth=2,min_size=1)
# # tree.fit(df,target='target')
# # y_0 = tree.predict(data_point)from mpi4py import MPI
# # print("DecisionTreeClassifier: ",y_0)from mpi4py import MPI
# ############################################################"""
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
forest = RandomForestClassifier(n_trees=5,max_depth=5, min_size=1)
trees = forest.fit(df, target='target', test = data_test)
if rank==0:
	print("predictions", trees)


# y_1 = forest.predict(data_point, trees)
# y_1 = max(set(trees), key=trees.count)
# print("RandomForestClassifier: ",y_1)
