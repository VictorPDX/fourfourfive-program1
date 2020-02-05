#!/usr/bin/env python3


import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import time
import math
import sys


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
	



def test(data, weights, targets):
	prediction_list = []
	correct = 0
	num_rows = data.shape[0]

	activations = np.dot(data, weights)
	
	for r in range(0, num_rows):
		tk = targets[r]
		yk = np.argmax(activations[r])
		prediction_list.append(yk)
		# print("%d) Firing perceptron %d"% (i, yk) )     
		if tk == yk:
			correct += 1
			# print("Correct: %d"% correct)
	accuracy = (correct / num_rows) 
	
	return accuracy, prediction_list



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
	data = data / 255
	
	# make a bias for every row
	bais = np.ones([rows , 1], dtype=float)

	# prepend the bias to the data
	# axis = 1 means vertical,   axis = 0 means horizontal
	data = np.concatenate((bais, data), axis=1)

	# return the data and the bias
	return data, labels

def add_bias(data, rows):
	# make a bias for every row
	bias = np.ones([rows , 1], dtype=float)

	# prepend the bias to the data
	# axis = 1 means vertical,   axis = 0 means horizontal
	return np.concatenate((bias, data), axis=1)

# rinse and repeat
def affine_projection(x, w):
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
	return add_bias(activations.reshape([1,rows]), activations.ndim)
	
def output_activation(x, w):
	# dot product 
	output = np.dot(x, w.T)
	# squash it and move on
	activation = 1 / (1 + np.exp(-output))
	return activation[0]

def error_function(layer, delta_k, delta_j, output_units, hidden_units, outputs, w_kj, targets, h_activations):

	delta_k = outputs * (1-outputs) * (targets - outputs)
	delta_j = h_activations * (1 - h_activations) * np.dot(w_kj, delta_k)
	
	#maybe we should add an index 0 for delta_k for the error to be at index 1   <-- I like this better
	# or remove index 0 from delta_j for it to be aligned with delta_k

	return delta_k, delta_j

def update_hidden_weights(eta, delta_k, hidden_activations, alpha, delta_W_kj, k, t):
	# alpha is momentum
	noh = eta * delta_k * hidden_activations
	momentum = alpha * delta_W_kj[t-1]

	# momentum should t-1 so new delta_W_ji should be one index greater
	delta_W_kj[t] = noh + momentum
	
	return delta_W_kj

def update_input_weights(eta, delta_j, inputs, alpha, delta_W_ji, hidden_units, j, t):
	# delta_j should start at 1
	unit = 1
	nox  = inputs * delta_j*eta
	momentum = alpha * delta_W_ji[t-1]
	# nox  = inputs[0] * delta_j[unit-1]*eta
	# momentum = alpha * delta_W_ji[t-1][unit-1]

	# momentum should t-1 so new delta_W_ji should be one index greater
	delta_W_ji[t][unit-1] = nox + momentum

	return delta_W_ji



def main():
	epochs = 70
	eta = .2
	batch = 1
	layers = 2
	hidden_units = int(sys.argv[1])
	output_units = 1
	alpha = 0.9

	
	
	# files
	train_data_file = 'data/train-images.idx3-ubyte'
	train_label_file = 'data/train-labels.idx1-ubyte'	
	test_data_file   = 'data/t10k-images.idx3-ubyte'
	test_labels_file = 'data/t10k-labels.idx1-ubyte'
	
	if __debug__:
		train_data = np.array([[1, 0], [0, 1]], dtype=float)
		train_label	= np.array([[.9], [-0.3]], dtype=float)
		
		train_data = add_bias(train_data, train_data.ndim)
	else:
		train_data, train_label = read_data_normalize_and_add_bias(train_label_file, train_data_file)
		test_data, test_labels = read_data_normalize_and_add_bias(test_labels_file, test_data_file)

	
	# test_rows, test_cols = test_data.shape
	# train_rows, train_cols = train_data.shape

	input_units = train_data.shape[1]

	if __debug__:
		# for testing purposes only
		# np.random.seed(1)
		# weights = np.random.randint(-5, 5, [train_cols, 10])
		weights = np.ones([hidden_units, input_units])
		hidden_weights = weights / 10
		weights = np.ones([output_units,hidden_units+1])
		output_weights = weights / 10
	else:
		# we need 10 sets of weigts, one for each digit we want to id
		weights = np.random.default_rng().uniform(-0.05, 0.05, [train_cols, 10])
	

	# when identifies digit 0 we get the zeroth row and one-hot matrix
	targets = np.identity(10, dtype=int)
	targets = np.where(targets > 0, 0.9, 0.1)


	train_accuracy_list = []
	test_accuracy_list  = []
	hidden_layers = np.zeros([layers], dtype=float)
	delta_W_kj = np.zeros([input_units, output_units, hidden_units+1])
	delta_W_ji = np.zeros([input_units, hidden_units, input_units])
	delta_k   = np.zeros([1, output_units])
	delta_j   = np.zeros([1, hidden_units])

	# what is this used for?
	delta_W   =np.zeros([])
	t = 1
	k = 1
	j = 1
	for epoch in range(epochs):
		start = time.time()
		for layer in range(layers):
			hidden_activations = affine_projection(train_data[layer], hidden_weights)
			output_activations = output_activation(hidden_activations, output_weights)
			
			
			targets = train_label[layer]
			
			# back propagate
			if output_activations[0] != targets[0]:
				# calculate the error
				w_kj_no_nias = output_weights[0][1:]
				delta_k, delta_j = error_function(layer, delta_k, delta_j, output_units, hidden_units, output_activations, output_weights[0][1:].reshape(2,1), targets, hidden_activations[0][1:])
				
				# update the hidden-to-output weights
				delta_W_kj =  update_hidden_weights(eta, delta_k, hidden_activations, alpha, delta_W_kj, k, t)
				output_weights = output_weights.T + delta_W_kj[t]
				
				# update the input-to-hidden weights
				delta_W_ji =  update_input_weights(eta, delta_j, train_data, alpha, delta_W_ji, hidden_units, j, t)
				hidden_weights = hidden_weights + delta_W_ji[t]

				print("Need to do")




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