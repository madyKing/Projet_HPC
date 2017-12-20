from math import sqrt
from RandomForest import RandomForest
from DecisionTreeClassifier import DecisionTreeClassifier
from mpi4py import MPI
import os

import sys
sys.path.insert(0, '/home/mamady/Bureau/Cours3A/HPC/Projet RF/codeBase/Projet_HPC/TreeMethods')

our_trees = list()
class RandomForestClassifier (RandomForest):

	"""
	A random forest classifier that derives from the base class RandomForest.

	:Attributes:
		**n_trees** (int) : The number of trees to use.

		**max_depth** (int): The maximum depth of tree.

		**min_size** (int): The minimum number of datapoints in terminal nodes.

		**cost_function** (str) : The name of the cost function to use: 'gini'.

		**trees** (list) : A list of the DecisionTree objects.

		**columns** (list) : The feature names.
	"""
	# global our_trees = list()
	def __init__(self, n_trees=10, max_depth=2, min_size=2, cost='gini'):
		"""
		Constructor for random forest classifier. This mainly just initialize
		the attributes of the class by calling the base class constructor.
		However, here is where it is the cost function string is checked
		to make sure it either using 'gini', otherwise an error is thrown.

		Args:
			cost (str) : The name of the cost function to use for evaluating
						 the split.

			n_trees (int): The number of trees to use.

			max_depth (int): The maximum depth of tree.

			min_size (int): The minimum number of datapoints in terminal nodes.

		"""
		# rank = MPI.COMM_WORLD.Get_rank()
		# if rank == 0:
		if cost != 'gini':
			raise NameError('Not valid cost function')
		else:
			RandomForest.__init__(self, cost,  n_trees=10, max_depth=2, min_size=2)



	def fit(self, train, target=None, test=None):#, predict=None):
		"""
		Fit the random forest to the training set train.  If a test set is provided
		then the return value wil be the predictions of the RandomForest on the
		test set.  If no test set is provide nothing is returned.


		Note: Below we set the number of features to use in the splitting to be
		the square root of the number of total features in the dataset.

		:Parameters:
			**train** (list or `Pandas DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_) : The training set.

			**target** (str or None) : The name of the target variable

			**test** (list or `Pandas DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_) : The test set.

		:Returns:
			(list or None): If a test set is provided then the return value wil be
			the predictions of the RandomForest on the test set.  If no test set
			is provide nothing is returned.
		"""
		# set the number of features for the trees to use.
		if isinstance(train, list) is False:
			if target is None:
				raise ValueError('If passing dataframe need to specify target.')
			else:

				train = self._convert_dataframe_to_list(train, target)

		n_features = int(sqrt(len(train[0])-1))


		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()
		size = comm.Get_size()

		if rank == 0:
			#	shuffle the initial data set
			for i in range(1, size):
				sample = self._subsample(train)
				comm.send(sample, dest = i, tag = 0)
			#	collect the trees from workers
			for i in range(1, size):
				masterTree = comm.recv(source = i, tag = 1)
				self.trees.append(masterTree)

			# if the test set is not empty then return the predictions
			if test is not None:
				if isinstance(test, list) is False:
					test = test.tolist()
				predictions = [self.predict(row) for row in test]

				return predictions

		else:
			smpl = comm.recv(source = 0, tag = 0)
			tree = DecisionTreeClassifier(self.max_depth, self.min_size, self.cost_function)
			tree.fit(smpl, n_features)
			comm.send(tree, dest = 0, tag = 1)

	def predict(self, row):
		"""
		Peform a prediction for a sample data point by bagging
		the prediction of the trees in the ensemble. The majority
		target class that is chosen.

		:Parameter: **row** (list or `Pandas Series <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html>`_ ) : The data point to classify.

		:Returns: (int) : The predicted target class for this data point.
		"""
		# rank = MPI.COMM_WORLD.Get_rank()
		# if rank ==0 :
		if isinstance(row, list) is False:
			row = row.tolist()
			predictions = [tree.predict(row) for tree in self.trees]
		else:
			predictions = [tree.predict(row) for tree in self.trees]
			# count retourne le nombre de fois que obj apparait dans la liste : list.count(obj)
		return max(set(predictions), key=predictions.count)




	def KFoldCV(self, dataset, target, n_folds=10):
		"""
		Perform k-fold cross validatation on the dataset
		and return the acrruracy of each training.

		:Parameters:
			**dataset** (list or `Pandas DataFrame <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html>`_) : The dataset in list form.

			** target ** (str) : The target variable name.

			**n_fold** (int) : The number of folds in the k-fold CV.

		:Returns: (list) : List of the accuracy of each Random Forest on each
			of the folds.
		"""
		if isinstance(dataset, list) is False:
			if target is None:
				raise ValueError('If passing dataframe need to specify target.')
			else:
				dataset = self._convert_dataframe_to_list(dataset, target)

		folds = self._cross_validation_split(dataset, n_folds)
		scores = list()
		for fold in folds:
			train_set = list(folds)
			train_set.remove(fold)
			train_set = sum(train_set, [])
			test_set = list()
			for row in fold:
				row_copy = list(row)
				test_set.append(row_copy)
				row_copy[-1] = None
			predicted = self.fit(train_set, test_set)
			actual = [row[-1] for row in fold]
			accuracy = self._metric(actual, predicted)
			scores.append(accuracy)
		return scores


	def _metric(self, actual, predicted):
		"""
		Returns the accuracy of the predictions for now, extending it
		to include other metrics.

		Args:
			actual (list) : The actual target values.
			predicted (list) : The predicted target values.

		Returns:
			float.  The accuracy of the predictions.

		"""
		return self._accuracy(actual, predicted)

	def _accuracy(self, actual, predicted):
		"""
		Computes the accuracy by counting how many predictions were correct.

		Args:
			actual (list) : The actual target values.
			predicted (list) : The predicted target values.

		Returns:
			float.  The accuracy of the predictions.
		"""
		correct = 0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				correct += 1
		return correct / float(len(actual)) * 100.0

import pandas as pd
def main():
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
	# from mpi4py import MPI
	rank = MPI.COMM_WORLD.Get_rank()
	# trees = []
	forest = RandomForestClassifier(n_trees=5,max_depth=5, min_size=1)
	trees = forest.fit(df, target='target', test = data_test)

	if rank==0:
		print("predictions", trees)
		# MPI.Finalize()
		# os._exit(0)
import timeit
if __name__ == "__main__":
	t_s = timeit.default_timer()
	main()
	t_e = timeit.default_timer()
	if MPI.COMM_WORLD.Get_rank() == 0:
		print ("Total time %s" %(t_e - t_s))
	# y_1 = forest.predict(data_point, trees)
	# y_1 = max(set(trees), key=trees.count)
	# print("RandomForestClassifier: ",y_1)
