#!/usr/bin/env python3

import sys
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import time
import math
import sys


class two_layer_nueral_net():
	"""This is a class representing a Nueral Net and all the components associated to it
	"""

	def __init__(self, training_data_set, training_labels, test_data_set, test_labels, hidden_units=100, output_units=10, momentum=0.9, epochs=50, eta=0.1):
		"""[summary]
		
		Arguments:
			training_data_set  -- the data to train the NN
			training_labels  -- the classes for that data
			test_data_set  -- the data to be test on the trained data
			test_labels  -- the test label
		
		Keyword Arguments:
			hidden_units {int} -- the number of hidden units (default: {20})
			output_units {int} -- the number of output units (default: {10})
			momentum {float} -- the momentum (default: {0.9})
			epochs {int} -- how many epochs to train/test on (default: {20})
			eta {float} -- the learning rate (default: {0.1})
		"""
		# hyper parameters
		self.eta = eta
		self.y = []    
		self.alpha = momentum
		self.epochs = epochs

		# data with their respective labels
		self.train_data = training_data_set
		self.train_labels = training_labels
		self.test_data = test_data_set
		self.test_labels = test_labels

		# account for the units
		self.input_units = training_data_set.shape[1]
		self.hidden_units = hidden_units
		self.output_units = output_units

		# the dimensions of the data sets
		self.test_rows, self.test_cols = test_data_set.shape
		self.train_rows, self.train_cols = training_data_set.shape
		
		# These are probably duplicates as I already have the dimensions and the units
		self.num_train_inputs = self.train_rows
		self.num_train_labels = self.train_cols
		self.num_test_inputs = self.test_rows
		self.num_test_labels = self.test_cols

		# the hidden activations for the training and testing
		self.train_hidden_activations = np.ones([1, hidden_units+1])
		self.test_hidden_activations  = np.ones([1, hidden_units+1])
		
		# when identifies digit 0 we get the zeroth row and one-hot matrix
		self.targets = np.identity(output_units, dtype=float)
		self.targets = np.where(self.targets > 0, 0.9, 0.1)
	
		# for testing, it is good to have static data every run
		# np.random.seed(1)
		# self.hidden_weights = np.random.randint(-5, 5, [hidden_units, self.input_units]) / 100
		# self.output_weights = np.random.randint(-5, 5, [output_units, hidden_units+1]) / 100
		
		# the weights , input-to-hidden and hidden to output
		self.hidden_weights = np.random.default_rng().uniform(-0.05, 0.05, [hidden_units, self.input_units])
		self.output_weights = np.random.default_rng().uniform(-0.05, 0.05, [output_units,hidden_units+1])
		
		# self.train_output_activations
		self.output_error = np.zeros([1, output_units])
		self.hidden_error = np.zeros([1, hidden_units])
		self.prediction_list = []

		# the change in weights
		self.delta_W_kj = np.zeros([self.output_units, self.hidden_units+1])
		self.delta_W_ji = np.zeros([self.hidden_units, self.input_units])

		# the accuracy for training and testing
		self.train_accuracy_list = []
		self.test_accuracy_list  = []
		
	# rinse and repeat
	def affine_projection(self, x, w):
		"""Performs an affine projection and activation function (sigmoid function)
		
		Arguments:
			x  -- the training data
			w  -- weights for forward phase
		
		Returns:
			np.array -- returns the hidden layer activations with the bias added to each hidden unit
		"""
		# dot product 
		# [1 x 785] @ [20 x 785].T
		output = np.dot(x, w.T)
		# squash it and move on
		activations = 1 / (1 + np.exp(-output))
		return activations
		
	def output_activation(self, x, w):
		"""Calculates the output activations
		
		Arguments:
			x  -- the hidden unit activations
			w  -- the hidden to output weights
		
		Returns:
			output_activation -- the output activations
		"""
		# dot product 
		output = np.dot(x, w.T)
		# squash it and move on
		output_activation = 1 / (1 + np.exp(-output))
		return output_activation

	def forward_phase(self, t, data, mode):
		"""Performed the affine projection and sigmoid (squashing) function
		
		Arguments:
			t  -- the data instance we are on
			data  -- the data being passed in
			mode  -- determines if it is a training or testing activations
		"""
		if mode == "train":
			self.train_hidden_activations[:,1:] = self.affine_projection(data, self.hidden_weights)
			self.train_output_activations       = self.output_activation(self.train_hidden_activations, self.output_weights)
		else:
			self.test_hidden_activations[:,1:] = self.affine_projection(data, self.hidden_weights)
			self.test_output_activations       = self.output_activation(self.test_hidden_activations, self.output_weights)
		



	def error_function(self, tk):
		""" Calculated the errof for the output activations and the hidden activations
			updates the output errors and the hidden errors

		Arguments:
			output_error  -- lowercase delta k
			hidden_error  -- lowercase delta j
			output_units  -- number of output units (k)
			hidden_units  -- number of hidden units (j)
			outputs  -- output activations
			w_kj  -- hidden to output weights without the bias  for hidden error since j starts at 1
			targets  -- labels
			h_activations  -- hidden actications

		
		"""
		self.output_error = self.train_output_activations * (1-self.train_output_activations) * (self.targets[tk] - self.train_output_activations)
		self.hidden_error = self.train_hidden_activations[:, 1:] * (1 - self.train_hidden_activations[:, 1:]) * np.dot(self.output_error, self.output_weights[:, 1:])
		
		#maybe we should add an index 0 for output_error for the error to be at index 1   <-- I like this better
		# or remove index 0 from hidden_error for it to be aligned with output_error


	def update_hidden_weights(self):
		""" Updates the hidden to output weights
			and stores it in the change in weights for output to hidden (kj)
		"""
		# alpha is momentum
		noh = self.eta * (self.train_hidden_activations.T @ self.output_error)
		momentum = self.alpha * self.delta_W_kj

		# momentum should t so new delta_W_ji should be one index greater
		self.delta_W_kj = noh.T + momentum
		self.output_weights += self.delta_W_kj
		
		# return delta_W_kj

				# delta_W_ji =  update_input_weights(input_units, eta, hidden_error, train_data[t], alpha, delta_W_ji, hidden_units, t)

	def update_input_weights(self, t):
		"""Update the input to hidden weights
		  and stores it in the change in weight for hidden to input (ji)
		
		Arguments:
			t  -- the index of the current data instance 
		
		"""
		# hidden_error should start at 1
		nox = self.eta * (self.train_data[t].reshape(self.train_data[t].shape[0], 1) @ self.hidden_error)
		momentum = self.alpha * self.delta_W_ji

		self.delta_W_ji = nox.T + momentum
		self.hidden_weights += self.delta_W_ji

		# return delta_W_ji


	def do_training(self, epoch):
		"""Performs the backpropagation for the NN
		
		Arguments:
			epoch  -- the current epoch
	"""
		correct = 0

		# t is one row of data from the training set
		for t in range(self.train_rows):
			self.forward_phase(t, self.train_data[t], "train")
			prediction = np.argmax(self.train_output_activations)
			# target = train_label[t]
			target = self.train_labels[t]
			
			if prediction == target:
				correct += 1
				
			# back propagate
			elif epoch > 0 and  prediction != target:
				# calculate the error
				self.error_function(target)
				self.update_hidden_weights()
				self.update_input_weights(t)
			
		# get the accuracy for every epoch
		train_accuracy = correct / self.num_train_inputs * 100
		self.train_accuracy_list.append(train_accuracy)



	def train(self):
		"""Runs the training and testing of the data one epoch at a time
		"""
		
		for epoch in range(self.epochs):

			start = time.time()
			self.do_training(epoch)
			self.test(epoch)
			# lapse_time = time.time() - start
			# print("\nEpoch", epoch, ")  lasted %0.2f sec"% lapse_time)
			# print("training accuracy: %0.2f"% self.train_accuracy_list[epoch])
			# print("testing accuracy: %0.2f"%  self.test_accuracy_list[epoch])

		# return train_accuracy_list



	def test(self, epoch):
		"""Runs the test data through the trained NN
		
		Arguments:
			epoch  -- the current epoch
		"""
		correct = 0
		
		for r in range(self.test_rows):
			target = self.test_labels[r]

			self.forward_phase(r, self.test_data[r], "test")
			prediction = np.argmax(self.test_output_activations)
			if epoch == self.epochs-1:
				self.prediction_list.append(prediction)

			if prediction == target:
				correct += 1
				
				# print("Correct: %d"% correct)
		test_accuracy = (correct / self.test_rows) * 100
		
		# return accuracy, prediction_list
		self.test_accuracy_list.append(test_accuracy)

	


	def confMat(self, epochs, filename):
		"""The confusion Matrix, representing the predicted to expecte data
		
		Arguments:
			epochs  -- The number of epoch that ran
			filename  -- the filename where to data the matrix in a csv format
		"""
		# create conf matrix
		conf = np.zeros([10, 10], dtype=int)
		correct = 0

	
		for i in range(self.test_rows):
			expected = self.test_labels[i]
			
			
			prediction = self.prediction_list[i]
		
			if expected == prediction:
				correct += 1

			conf[expected, prediction] += 1

		print(conf)
		print("Accuracy: %0.2f"% ((correct /self.test_rows) * 100))
		print("Leraning rate:", self.eta)
		print("# hidden units: ", self.hidden_units)
		print("momentum: ", self.alpha)
		print("Epochs:", epochs)
		print("Total: ", np.sum(conf))
		# print("Batch size:", batch)

		np.savetxt(filename + ".csv", conf, delimiter=',')


###########    END OF CLASS  ###############



def accuracyGraph(train_accuracy_list, test_accuracy_list, file_name):
	"""Prints the graph showing the accuracy for every epoch
	
	Arguments:
		train_accuracy_list  -- the accuracy of the training set for each epoch
		test_accuracy_list  -- the accuracy of the testing set for each epoch
		file_name  -- the name of the file that will be saved
	"""
	plt.title("Accuracy Graph")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy (%)")

	
	plt.plot(train_accuracy_list, label="Training")
	plt.plot(test_accuracy_list,  label="Test")
	plt.legend()
	plt.savefig(file_name +".png")
	# plt.show()





def read_data_normalize_and_add_bias(classes, features):
	"""Read in the file with classes (labels) and the features for each class
	Normalizes the features and prepends a bias to each data instance
	
	Arguments:
		classes  -- the labels
		features  -- the features for each class
	
	Returns:
		The normalized data with a bias and the the labels
	"""
	
	data, labels = loadlocal_mnist(
		images_path=features, 
		labels_path=classes)
	rows = data.shape[0]

	# normalize
	data = add_bias(data / 255, rows)
	
	# # make a bias for every row
	# bais = np.ones([rows , 1], dtype=float)

	# # prepend the bias to the data
	# # axis = 1 means vertical,   axis = 0 means horizontal
	# data = np.concatenate((bais, data), axis=1)

	# return the data and the bias
	return data, labels

def add_bias(data, rows):
	"""Prepends a bias to any data passed in
	
	Arguments:
		data  -- The data needing a bias
		rows  -- the number of rows that need a bias
	
	Returns:
		The data now with a bias
	"""
	# make a bias for every row
	bias = np.ones([rows , 1])

	# prepend the bias to the data
	# axis = 1 means vertical,   axis = 0 means horizontal
	return np.concatenate((bias, data), axis=1)







def main():
	output_units = 10
	hidden_units = int(sys.argv[1])
	
	# files
	train_data_file = 'data/train-images.idx3-ubyte'
	train_label_file = 'data/train-labels.idx1-ubyte'	
	test_data_file   = 'data/t10k-images.idx3-ubyte'
	test_labels_file = 'data/t10k-labels.idx1-ubyte'
	
	train_data, train_labels = read_data_normalize_and_add_bias(train_label_file, train_data_file)
	test_data, test_labels = read_data_normalize_and_add_bias(test_labels_file, test_data_file)

	



	eta = .1
	alpha = 0.9

	print("Experiment 1:")
	
	epochs = 50
	hidden_units = 20
	# creates a new NN with 20 hidden units 
	nn_1_20_hidden_units = two_layer_nueral_net(train_data, train_labels, test_data, test_labels, hidden_units, output_units, alpha, epochs)
	start = time.time()
	nn_1_20_hidden_units.train()
	lapse_time = time.time() - start

	fname = "nn_1_20_hidden_units_hidden_units"
	# prints accuracy graph for training and test data
	accuracyGraph(nn_1_20_hidden_units.train_accuracy_list, nn_1_20_hidden_units.test_accuracy_list, fname)

	# print confusion matrix for the test data
	nn_1_20_hidden_units.confMat(epochs, fname )
	print("TOTAL TIME: %0.2f sec\n\n\n"%lapse_time)





	hidden_units = 50
	# creates a new model with 50 hidden units and with 50 epochs still
	nn_1_50_hidden_units = two_layer_nueral_net(train_data, train_labels, test_data, test_labels, hidden_units, output_units, alpha, epochs)
	start = time.time()
	nn_1_50_hidden_units.train()
	lapse_time = time.time() - start
	fname = "nn_1_50_hidden_units_hidden_units"
	# prints accuracy graph for training and test data
	accuracyGraph(nn_1_50_hidden_units.train_accuracy_list, nn_1_50_hidden_units.test_accuracy_list, fname)

	# prints confusion matrix for the test data
	nn_1_50_hidden_units.confMat(epochs, fname)
	print("TOTAL TIME: %0.2f sec\n\n\n"%lapse_time)





	hidden_units = 100
	#creates a new NN with 100 hidden units
	nn_1_100_hidden_units = two_layer_nueral_net(train_data, train_labels, test_data, test_labels, hidden_units, output_units, alpha, epochs)
	start = time.time()
	nn_1_100_hidden_units.train()
	lapse_time = time.time() - start
	fname = "nn_1_100_hidden_units_hidden_units"

	# prints accuracy graph for training and test data
	accuracyGraph(nn_1_100_hidden_units.train_accuracy_list, nn_1_100_hidden_units.test_accuracy_list, fname)

	# print confusion matrix for the test data
	nn_1_100_hidden_units.confMat(epochs, fname)
	print("\n\nTOTAL TIME: %0.2f sec\n\n\n"%lapse_time)
	
	




	print("Experiment 2:")
	alpha = 0.0
	# a NN without any momentum
	nn_2_momentum_0 = two_layer_nueral_net(train_data, train_labels, test_data, test_labels, hidden_units, output_units, alpha, epochs)
	start = time.time()
	nn_2_momentum_0.train()
	lapse_time = time.time() - start
	fname = "nn_2_momentum_0"

	# prints accuracy graph for training and test data
	accuracyGraph(nn_2_momentum_0.train_accuracy_list, nn_2_momentum_0.test_accuracy_list, fname)

	# print confusion matrix for the test data
	nn_2_momentum_0.confMat(epochs, fname)
	print("\n\nTOTAL TIME: %0.2f sec\n\n\n"%lapse_time)
	



	# momentum is 0.25
	alpha = 0.25
	nn_2_momentum_0_25 = two_layer_nueral_net(train_data, train_labels, test_data, test_labels, hidden_units, output_units, alpha)
	start = time.time()
	nn_2_momentum_0_25.train()
	lapse_time = time.time() - start
	fname = "nn_2_momentum_0_25"

	# prints accuracy graph for training and test data
	accuracyGraph(nn_2_momentum_0_25.train_accuracy_list, nn_2_momentum_0_25.test_accuracy_list, fname)

	# printf confusion matrix for the test data
	nn_2_momentum_0_25.confMat(epochs, fname)
	print("\n\nTOTAL TIME: %0.2f sec\n\n\n"%lapse_time)
	





	alpha = 0.5
	nn_2_momentum_0_50 = two_layer_nueral_net(train_data, train_labels, test_data, test_labels, hidden_units, output_units, alpha)
	start = time.time()
	nn_2_momentum_0_50.train()
	lapse_time = time.time() - start
	fname = "nn_2_momentum_0_50"

	# prints accuracy graph for training and test data
	accuracyGraph(nn_2_momentum_0_50.train_accuracy_list, nn_2_momentum_0_50.test_accuracy_list, fname)

	# printf confusion matrix for the test data
	nn_2_momentum_0_50.confMat(epochs, fname)
	print("\n\nTOTAL TIME: %0.2f sec\n\n\n"%lapse_time)
	




	print("Experiment 3:")
	
	# load a fresh set of data
	raw_data, raw_labels = loadlocal_mnist(
		images_path=train_data_file, 
		labels_path=train_label_file)
	# put the labels with the features before we shuffle
	full_data = np.concatenate((raw_labels.reshape(raw_labels.shape[0], 1), raw_data), axis=1)
	# shuffle
	np.random.shuffle(raw_data)
	
	fname = "nn_3_sample_0_25"
	# batch size
	batch = 0.25
	size = int(batch * raw_data.shape[0])
	
	# get the features columns 1 and rest, normalize,  and add "size" bias
	raw_data_2 = add_bias(full_data[:size, 1:]/255, size)
	# get the lables only column 0
	raw_labels_2 = full_data[:size, 0:1]
	test_data, test_labels = read_data_normalize_and_add_bias(test_labels_file, test_data_file)


	
	# create a new NN model
	nn_3_sample_0_25 = two_layer_nueral_net(raw_data_2, raw_labels_2, test_data, test_labels)
	start = time.time()
	# and make it work
	nn_3_sample_0_25.train()
	lapse_time = time.time() - start


	# # prints accuracy graph for training and test data
	accuracyGraph(nn_3_sample_0_25.train_accuracy_list, nn_3_sample_0_25.test_accuracy_list, fname)

	# # prints confusion matrix for the test data
	nn_3_sample_0_25.confMat(epochs, fname)
	print("\n\nTOTAL TIME: %0.2f sec\n\n\n"%lapse_time)
	



	
	
	batch = 0.5
	fname = "nn_3_sample_0_5"
	size = int(batch * raw_data.shape[0])
	# randomise the new data that we have not worked on
	np.random.shuffle(full_data)
	
	
	# get the features columns 1 and rest, normalize,  and add "size" bias
	raw_data_3 = add_bias(full_data[:size, 1:]/255, size)
	# get the lables only column 0
	raw_labels_3 = full_data[:size, 0:1].astype(int)
	# create a new NN with the new data, there are default parameters in use here
	nn_3_sample_0_50 = two_layer_nueral_net(raw_data_3, raw_labels_3, test_data, test_labels)
	start = time.time()
	# let the fun begin
	nn_3_sample_0_50.train()
	lapse_time = time.time() - start


	# # prints accuracy graph for training and test data
	accuracyGraph(nn_3_sample_0_50.train_accuracy_list, nn_3_sample_0_50.test_accuracy_list, fname)

	# # printf confusion matrix for the test data
	nn_3_sample_0_50.confMat(epochs, fname)
	print("\n\nTOTAL TIME: %0.2f sec\n\n\n"%lapse_time)
	






if __name__ == "__main__":
    main()
