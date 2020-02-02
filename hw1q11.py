


import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import time


def one_hot_encoded_Targets(target_list, rows):
	targets = np.zeros([rows, 10])
	for i in range(target_list.shape[0]):
		targets[i][target_list[i]] = 1
	return targets


def train(epoch, eta, data, labels, weights, targets):
	correctList = []
	
	correct = 0
	wrong = 0
	rows, cols = data.shape
	for i in range(0, rows):
		# I need to check one row of data at a time to see if weights need to be updated
		# subseuent rows of data will fire with the new weights

		#                  [1x785] @ [785x10]
		activations = np.dot(data[i], weights)
		tk = labels[i]
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
				delta_w = eta * np.outer(data[i],   delta)   # [785x 10]
				weights = weights + delta_w
				wrong += 1

				# print("AFTER ", weights)
	accuracy = correct / data.shape[0] 
	# print('\n weight and new weight are same: ', np.array_equal(hold_weight, weights))
	
	
	# if epoch > 0 and abs(correctList[epoch] - correctList[epoch - 1]) < 0.01:	
	# 	return (correctList, weights, epochList, weightsList)
	return weights, accuracy
	# return accuracy




def test(data, weights, targets):
	prediction_list = []
	correct = 0
	rows = data.shape[0]

	activations = np.dot(data, weights)
	output = np.where(activations > 0, 1, 0)
	
	for i in range(0, rows):
		tk = targets[i]
		yk = np.argmax(activations[i])
		prediction_list.append(yk)
		# print("%d) Firing perceptron %d"% (i, yk) )     
		if tk == yk:
			correct += 1
			# print("Correct: %d"% correct)


	accuracy = (correct / rows) 
	
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
	target_cols = len(targets)
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
	rows, cols = data.shape

	# normalize
	data = data / 255
	
	# make a bias for every row
	bais = np.ones([rows , 1], dtype=float)

	# prepend the bias to the data
	# axis = 1 means vertical,   axis = 0 means horizontal
	data = np.concatenate((bais, data), axis=1)

	# return the data and the bias
	return data, labels



def main():
	epochs = 70
	eta = .1
	batch = 1
	

	train_data_file = 'data/train-images.idx3-ubyte'
	train_label_file = 'data/train-labels.idx1-ubyte'	
	test_data_file   = 'data/t10k-images.idx3-ubyte'
	test_labels_file = 'data/t10k-labels.idx1-ubyte'
	

	train_data, train_label = read_data_normalize_and_add_bias(train_label_file, train_data_file)
	test_data, test_labels = read_data_normalize_and_add_bias(test_labels_file, test_data_file)

	
	test_rows, test_cols = test_data.shape
	train_rows, train_cols = train_data.shape

	# we need 10 sets of weigts, one for each digit we want to id
	weights = np.random.default_rng().uniform(-0.05, 0.05, [train_cols, 10])

	# TODO
	# FIXME for testing purposes only
	np.random.seed(1)
	weights = np.random.randint(-5, 5, [train_cols, 10])
	weights = weights / 100

	# when identifies digit 0 we get the zeroth row and one-hot matrix
	targets = np.identity(10, dtype=int)



	train_accuracy_list = []
	test_accuracy_list  = []
	
	for epoch in range(epochs):
		start = time.time()
		# train, update weights and  check training accuracy
		weights, train_accuracy = train(epoch, eta, train_data, train_label, weights, targets)
		train_accuracy_list.append(train_accuracy)
		# test and check test accuracy
		print('\nEpoch: %d\nTraining accuracy %0.2f' % (epoch, train_accuracy))
	
		#                     data      weights    targets
		test_accuracy, predictions = test(test_data, weights, test_labels)
		test_accuracy_list.append(test_accuracy)
		# prediction_list.append(prediction)

		print('Test accuracy     %0.2f' % (test_accuracy))
		lapse_time = time.time() - start
		print("Epoch %d lasted %0.2f seconds"% (epoch, lapse_time))





	# prints accuracy graph for training and test data
	accuracyGraph(train_accuracy_list, test_accuracy_list)

	# printf confusion matrix for the test data
	confMat(test_data, test_labels, epochs, eta, predictions)



if __name__ == "__main__":
    main()