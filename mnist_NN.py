#!/usr/bin/env python3

import sys

print(sys.version)

import numpy as np
from mlxtend.data import loadlocal_mnist
# import matplotlib.pyplot as plt
import time
import math
import sys


class two_layer_nueral_net():

	def __init__(self, training_data_set, training_labels, test_data, test_labels, hidden_units=20, output_units=10, momentum=0.9, epochs=20, eta=0.1):
		self.eta = eta
		self.y = []    
		self.alpha = momentum

		self.train_data = training_data_set
		self.train_labels = training_labels
		self.input_units = training_data_set.shape[1]
		self.hidden_units = hidden_units
		self.output_units = output_units

		self.test_rows, self.test_cols = test_data.shape
		self.train_rows, self.train_cols = training_data_set.shape
		
		self.num_train_inputs = self.train_rows
		self.num_train_labels = self.train_cols
		self.num_test_inputs = self.test_rows
		self.num_test_labels = self.test_cols

		np.random.seed(1)
		self.hidden_weights = np.random.randint(-5, 5, [hidden_units, self.input_units]) / 100
		self.output_weights = np.random.randint(-5, 5, [output_units, hidden_units+1]) / 100
		
		# self.output_activations
		self.output_error = np.zeros([1, output_units])
		self.hidden_error = np.zeros([1, hidden_units])
		

		
		self.delta_W_kj = np.zeros([self.input_units, output_units, hidden_units+1])
		self.delta_W_ji = np.zeros([self.input_units, hidden_units, self.input_units])

		self.train_accuracy_list = []
		self.test_accuracy_list  = []
		
	# rinse and repeat
	def affine_projection(self, x, w):
		"""Performs an affine projection and activation function (sigmoid function)
		
		Arguments:
			x {np array} -- the training data
			targets {np array} -- training targets
			w {np array} -- weights for forward phase
		
		Returns:
			np.array -- returns the hidden layer activations with the bias added to each hidden unit
		"""
		# dot product 
		output = np.dot(x, w.T)
		# squash it and move on
		activations = 1 / (1 + np.exp(-output))
		rows = activations.shape[0]
		
		bias = np.ones([rows , 1], dtype=float)
			
		# prepend the bias to the data
		# axis = 1 means vertical,   axis = 0 means horizontal
		return np.concatenate((bias, activations), axis=1)
		
	def output_activation(self, x, w):
		# dot product 
		output = np.dot(x, w.T)
		# squash it and move on
		output_activation = 1 / (1 + np.exp(-output))
		return output_activation

	def forward_phase(self, t, data):
		self.hidden_activations = self.affine_projection(data[t-1], self.hidden_weights)
		self.output_activations = self.output_activation(self.hidden_activations, self.output_weights)
		


		# output_error, hidden_error = nn.error_function( output_units, hidden_units, output_activations, output_weights[:, 1:], targets[target], hidden_activations[:, 1:])


	def error_function(self, targets):
		"""[summary]

		Arguments:
			output_error {[type]} -- lowercase delta k
			hidden_error {[type]} -- lowercase delta j
			output_units {[type]} -- number of output units (k)
			hidden_units {[type]} -- number of hidden units (j)
			outputs {[type]} -- output activations
			w_kj {[type]} -- hidden to output weights without the bias  for hidden error since j starts at 1
			targets {[type]} -- labels
			h_activations {[type]} -- hidden actications

		Returns:
			[type] -- the output errors and the hidden errors
		"""
		self.output_error = self.output_activations * (1-self.output_activations) * (targets - self.output_activations)
		self.hidden_error = self.hidden_activations[:, 1:] * (1 - self.hidden_activations[:, 1:]) * np.dot(self.output_error, self.output_weights[:, 1:])
		
		#maybe we should add an index 0 for output_error for the error to be at index 1   <-- I like this better
		# or remove index 0 from hidden_error for it to be aligned with output_error


	def update_hidden_weights(self, t):
		# alpha is momentum
		noh = self.eta * (self.hidden_activations.T @ self.output_error)
		momentum = self.alpha * self.delta_W_kj[t-1]

		# momentum should t-1 so new delta_W_ji should be one index greater
		self.delta_W_kj[t] = noh.T + momentum
		self.output_weights += self.delta_W_kj[t]
		
		# return delta_W_kj

				# delta_W_ji =  update_input_weights(input_units, eta, hidden_error, train_data[t-1], alpha, delta_W_ji, hidden_units, t)

	def update_input_weights(self, t):
		# hidden_error should start at 1
		nox = self.eta * (self.train_data[t-1].reshape(self.train_data[t-1].shape[0], 1) @ self.hidden_error)
		momentum = self.alpha * self.delta_W_ji[t-1]

		self.delta_W_ji[t] = nox.T + momentum
		self.hidden_weights += self.delta_W_ji[t]

		# return delta_W_ji



	def train(self):
		
		
		# what is this used for?
		# delta_W   =np.zeros([])
		# j = 1
		# k = 1
		correct = 0
		for epoch in range(epochs):

			start = time.time()
			
			
			# t is one row of data from the training set
			for t in range(1, self.train_rows+1):
				self.forward_phase(t, self.train_data[t-1])
				# hidden_activations = affine_projection(train_data[t-1], hidden_weights)
				# output_activations = output_activation(hidden_activations, output_weights)
				prediction = np.argmax(self.output_activations)
				# target = train_label[t-1]
				target = self.train_labels[t-1]
				
				if prediction == target:
					correct += 1
				# back propagate
				elif epoch > 0 and  prediction != target:
					# calculate the error
					#                                                                                                                          output weights no bias vertical vector        label       hidden units no bias
					self.error_function(targets[target])
					self.update_hidden_weights(t)
					self.update_input_weights(t)
					# # update the hidden-to-output weights
					# delta_W_kj =  update_hidden_weights(eta, output_error, hidden_activations, alpha, delta_W_kj, k, t)
					# output_weights = output_weights + delta_W_kj[t]
					
					# # update the input-to-hidden weights  one input at  a time
					# delta_W_ji =  update_input_weights(input_units, eta, hidden_error, train_data[t-1], alpha, delta_W_ji, hidden_units, t)
					# hidden_weights = hidden_weights + delta_W_ji[t]

				# get the accuracy for every epoch
				train_accuracy = correct / self.num_train_inputs * 100
			self.train_accuracy_list.append(train_accuracy)
		# return train_accuracy_list



	def test(self, weights, targets):
		prediction_list = []
		correct = 0
		# num_rows = data.shape[0]

# this should be forward phase!!!!
		activations = np.dot(self.num_test_inputs, self.hidden_weights)
		
		for r in range(self.test_rows):
			tk = targets[r]
			yk = np.argmax(activations[r])
			prediction_list.append(yk)
			# print("%d) Firing perceptron %d"% (i, yk) )     
			if tk == yk:
				correct += 1
				# print("Correct: %d"% correct)
		accuracy = (correct / self.test_rows) 
		
		return accuracy, prediction_list




def one_hot_encoded_Targets(target_list, rows):
	targets = np.zeros([rows, 10])
	for i in range(target_list.shape[0]):
		targets[i][target_list[i]] = 1
	return targets


def train(epoch, eta, data, labels, weights, targets):
	correct = 0
	wrong = 0
	num_rows = data.shape[0]
	for r in range(num_rows):
		# I need to check one row of data at a time to see if weights need to be updated
		# subseuent rows of data will fire with the new weights

		#                  [1x785] @ [785x10]
		activations = np.dot(data[r], weights)
		tk = labels[r]
		yk = np.argmax(activations)
		output = np.where(activations > 0, 1, 0)
		
		# print("%d) Firing perceptron %d"% (i, yk) )

		if tk == yk:
			correct +=1
			# print("Correct: %d"% correct)


		elif epoch != 0 and  tk != yk:
				# print("Activations: ", activations)
				# print("tk = ", tk)
				# print("Perceptron %d fired"%yk)
				# print("Before: ", weights)
				delta = (targets[tk] - output)  # [1x10]
				#output_k = output.reshape(10,1)
				#target_k = targets[tk].reshape(10,1)
				# data_k = data[i]  # [1x785]

										# [1x785] @ [1x10]
				delta_w = eta * np.outer(data[r],   delta)   # [785x 10]
				weights = weights + delta_w
				wrong += 1

				# print("AFTER ", weights)
	accuracy = correct / num_rows 
	# print('\n weight and new weight are same: ', np.array_equal(hold_weight, weights))
	
	return weights, accuracy
	






def accuracyGraph(correctList, test_accuracy_list):
	plt.title("Accuracy Graph")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy (%)")

	
	plt.plot(correctList, label="Training")
	plt.plot(test_accuracy_list, label="Test")
	plt.legend()
	# plt.show()
	plt.savefig("plotfile.png")



# confMat(test_data, test_labels, epochs, eta)
def confMat(data, targets, epochs, learning, prediction_list):
	# create conf matrix
	conf = np.zeros([10, 10], dtype=int)
	correct = 0

	data_rows = data.shape[0]

	for i in range(0, data_rows):
		expected = targets[i]
		prediction = prediction_list[i]
	
		if expected == prediction:
			correct += 1

		conf[expected, prediction] += 1

	print(conf)
	print("Accuracy:", (correct / data_rows) * 100, "%")
	print("Leraning rate:", learning)
	print("Epochs:", epochs)
	# print("Batch size:", batch)

	np.savetxt("confusion matrix.csv", conf, delimiter=',')



def read_data_normalize_and_add_bias(classes, features):
	
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
	# make a bias for every row
	bias = np.ones([rows , 1], dtype=float)

	# prepend the bias to the data
	# axis = 1 means vertical,   axis = 0 means horizontal
	return np.concatenate((bias, data), axis=1)







def main():
	eta = .1
	batch = 1
	alpha = 0.9
	epochs = 70
	output_units = 10
	hidden_units = int(sys.argv[1])
	
	
	# files
	train_data_file = 'data/train-images.idx3-ubyte'
	train_label_file = 'data/train-labels.idx1-ubyte'	
	test_data_file   = 'data/t10k-images.idx3-ubyte'
	test_labels_file = 'data/t10k-labels.idx1-ubyte'
	
	train_data, train_labels = read_data_normalize_and_add_bias(train_label_file, train_data_file)
	test_data, test_labels = read_data_normalize_and_add_bias(test_labels_file, test_data_file)

	
	# when identifies digit 0 we get the zeroth row and one-hot matrix
	targets = np.identity(output_units, dtype=float)
	targets = np.where(targets > 0, 0.9, 0.1)
	


	# train_rows = train_data.shape[0]
	# input_units = train_data.shape[1]
	# # if __debug__:
	# # 	# for testing purposes only
	# np.random.seed(1)
	# hidden_weights = np.random.randint(-5, 5, [hidden_units, input_units]) / 100
	# output_weights = np.random.randint(-5, 5, [output_units,hidden_units+1]) / 100
	# 	hidden_weights = np.ones([hidden_units, input_units]) / 10
	# 	output_weights = np.ones([output_units,hidden_units+1]) / 10
	# else:
	# 	# we need 10 sets of weigts, one for each digit we want to id
	# hidden_weights = np.random.default_rng().uniform(-0.05, 0.05, [hidden_units, input_units])
	# output_weights = np.random.default_rng().uniform(-0.05, 0.05, [output_units,hidden_units+1])

	print("Experiment 1:")
	epochs = 50
	hidden_units = 20
	nn_1_20 = two_layer_nueral_net(train_data, train_labels, test_data, test_labels, hidden_units, output_units, alpha, epochs)
	nn20.train()

	hidden_units = 50
	nn_1_50 = two_layer_nueral_net(train_data, train_labels, test_data, test_labels, hidden_units, output_units, alpha, epochs)
	nn_1_50.train()

	hidden_units = 100
	nn_1_100 = two_layer_nueral_net(train_data, train_labels, test_data, test_labels, hidden_units, output_units, alpha, epochs)
	nn_1_100.train()
	
	
	



		# train, update weights and  check training accuracy
		weights, train_accuracy = train(epoch, eta, train_data, train_label, weights, targets)
		train_accuracy_list.append(train_accuracy)
		if __debug__:
			print('\nEpoch: %d\nTraining accuracy %0.2f' % (epoch, train_accuracy))
	
		#                     data      weights    targets
		test_accuracy, predictions = test(test_data, weights, test_labels)
		test_accuracy_list.append(test_accuracy)
		# prediction_list.append(prediction)

		if __debug__:
			print('Test accuracy     %0.2f' % (test_accuracy))
			lapse_time = time.time() - start
			print("Epoch %d lasted %0.2f seconds"% (epoch, lapse_time))





	# prints accuracy graph for training and test data
	accuracyGraph(train_accuracy_list, test_accuracy_list)

	# printf confusion matrix for the test data
	confMat(test_data, test_labels, epochs, eta, predictions)



if __name__ == "__main__":
    main()

# def backprop(self):
# 	delta_hidden = np.zeros(np.shape(self.hidden_weights))
# 	delta_output = np.zeros(np.shape(self.output_weights))

# 	# for i in range(0, self.input.shape[1]):
# 	output = self.forward(self.input[0])
# 	output_error, hidden_error = self.error(output, 0)
# 	# need a 10x21 (if hidden_inputs = 20)
# 	delta_output = self.eta*(np.dot(self.hidden_inputs.T, output_error)).T + self.momentum * delta_output
# 	# need a 785 x 20 (exclude the bias from hidden)
# 	# TODO reshape_input = self.input[0].reshape(785,1)
# 	reshape_input = self.input[0].reshape(3,1)
# 	delta_hidden = self.eta*(np.dot(reshape_input, hidden_error[:, 1:])).T + self.momentum * delta_hidden

# 	self.output_weights += delta_output
# 	self.hidden_weights += delta_hidden


# 	  def error(self, output, labelIndex):
#     # def error(self, output):
#         t_k = self.labels[labelIndex]
#         # need [1x 10]
#         # TODO output_error = output*(1-output)*(self.target[t_k]-output)
#         output_error = output*(1-output)*(self.target-output)
#         # need [1 x 21] (if hidden layer = 20)
#         hidden_error = self.hidden_inputs*(1-self.hidden_inputs)*np.dot(output_error, self.output_weights)
#         return output_error, hidden_error