
# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

def readData(fname):
	with open(fname) as f:
		ncols = len(f.readline().split(','))
	data = np.loadtxt(fname, dtype=int, delimiter=",", usecols=range(1, ncols))
	
	# normalize the data
	data = data / 255
	

	bais = np.ones([data.shape[0] , 1], dtype=int)
	# axis = 1 means vertical,   axis = 0 means horizontal
	data = np.concatenate((bais, data), axis=1)
	return data

def one_hot_encoded_Targets(target_list, rows):
	targets = np.zeros([rows, 10])
	for i in range(target_list.shape[0]):
		targets[i][target_list[i]] = 1
	return targets

def trainings( epochs,  eta, data, labels, weights, targets, batch):
	weightsList = []
	weightsList.append(weights)
	correctList = []
	epochList = []
	correct = 0
	wrong = 0
	rows, cols = data.shape
	for epoch in range(0, epochs):
		correct = 0
		for i in range(0, rows):
			# I need to check one row of data at a time to see if weights need to be updated
			# subseuent rows of data will fire with the new weights

			#                  [1x785] @ [785x10]
			activations = np.dot(data[i], weights)
			tk = labels[i]
			yk = np.argmax(activations)
			output = np.where(activations > 0, 1, 0)
			
			

			# if epoch == 0:
			# 	if tk == yk:
			# 		correct +=1

			# else:
			if tk != yk:
				print("Activations: ", activations)
				print("tk = ", tk)
				print("Perceptron %d fired"%yk)
				print("Before: ", weights)
				delta = (targets[tk] - output)  # [1x10]
				#output_k = output.reshape(10,1)
				#target_k = targets[tk].reshape(10,1)
				# data_k = data[i]  # [1x785]

										# [1x785] @ [1x10]
				delta_w = eta * np.outer(data[i],   delta)   # [785x 10]
				weights = weights + delta_w
				wrong += 1

				print("AFTER ", weights)
			else:
				correct +=1
		accuracy = correct / data.shape[0] * 100 
		# print('\n weight and new weight are same: ', np.array_equal(hold_weight, weights))
		print('\n epoch: %s, accuracy %s' % (epoch, accuracy))
		correctList.append(accuracy)
		weightsList.append(weights)
		epochList.append(epoch)
		if epoch > 0 and abs(correctList[epoch] - correctList[epoch - 1]) < 0.01:	
			return (correctList, weights, epochList, weightsList)
	return (correctList, weights, epochList, weightsList)
	# return accuracy

def training(epochs, eta, data, labels, w, targets, batch):
	weightsList = []
	weightsList.append(w)
	correctList = []
	epochList = []
	correct = 0
	wrong = 0
	for epoch in range(0, epochs):
		print("Epoch:", (epoch + 1), "/", epochs)
		for i in range(0, data.shape[0]):
			activations = np.dot(data[i], w)  
			outputs = np.where(activations > 0, 1, 0)  
			yk = np.argmax(activations)
			tk = labels[i]

			w_transpose = w.T
			if yk == tk:
				correct += 1
			else:
				w_transpose[i] = w_transpose[i] + (eta *  (targets[i] - outputs) * data[i])
				wrong += 1
		percent = correct / batch * 100
		
		print("accuracy: %s \n" % str(percent))
		correctList.append(percent)
		weightsList.append(w)
		epochList.append(epoch)
	return (correctList, w, epochList, weightsList)



def train(epochs, eta, data, label, w, targets, batch):
	for epoch in range(0, epochs):
		correct = 0
		
		print("Epoch: {epoch} / {epochs}")
		i = 0
		for data_k in data:
			activations = np.dot(data_k, w)
			#threshold/activation function
			outputs = np.where(activations > 0, 1, 0)
			yk = np.argmax(activations)
			tk = label[i]
			# print("t^k = %d" % label[i])
			# print(targets[label[i]])
			# print("y = ")
			# print(activations)
			# print("yk = %s" % str(yk))
			
			# print("y = %s" % str(outputs))

			if yk != tk:
				# print('weights: ', w[:, yk])
				# delta_w = .01 * (tk-yk)* data[i]
				for row in range(0, w.shape[1]):
					for col in range(0, w.shape[0]):
						w[row][col] = w[row][col] +  eta * (targets[row] - outputs[row])* data[i][col]
					# w[:, yk] = w[:, yk] + delta_w
				# print('weights: ', w[:, yk])
			else:
				correct += 1
				
			
			i += 1
		print("accuracy = %s \n" % str(correct/data.shape[0]))
	print("this is the end of the epoch")




def test(epochs, data, weightList, targets, batch):
	correctList = []
	correct = 0
	for epoch in range(0, epochs):
		correct = 0
		activations = np.dot(data, weightList[epoch])
		output = np.where(activations > 0, 1, 0)
		
		for i in range(0, data.shape[0]):
			# expected = targets[i]
			# out = np.argmax(activations[i])
		
			if targets[i] == np.argmax(activations[i]):
				correct += 1
	
		accuracy = (correct / data.shape[0]) * 100 
		correctList.append(accuracy)
	return correctList



# helper function that graphs accuracy per epoch for both training
# and testing data
def accuracyGraph(correctList, epochList, testCorrect):
	plt.title("Accuracy Graph")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy (%)")
	plt.axis([0, len(epochList), 0, 100])

	
	plt.plot(epochList, correctList, label="Training")
	plt.plot(epochList, testCorrect, label="Test")
	plt.legend()
	plt.show()
	plt.savefig("plotfile.png")


# helper function that computes the confusion matrix for the testing
# data using the latest (most accurate) weights
# confMat(test_data, test_labels, final_weights, epochs, batch, eta)
def confMat(data, targets, weights, epochs, batch, learning):
	# create conf matrix
	target_cols = len(targets)
	conf = np.zeros([10, 10], dtype=int)
	corr = 0

	# for i in range(10):
	# 	conf[0, i + 1] = i
	# 	conf[i + 1, 0] = i
		
	data_rows, data_cols = data.shape
	activations = np.dot(data, weights)
	output = np.where(activations > 0, 1, 0)

	for i in range(0, data_rows):
		expected = targets[i]
		out = np.argmax(output[i])

		if expected == out:
			corr += 1

		conf[expected, out] += 1

	print(conf)
	print("Accuracy:", (corr / data_rows) * 100, "%")
	print("Leraning rate:", learning)
	print("Epochs:", epochs)
	print("Batch size:", batch)

	np.savetxt("confusion matrix.csv", conf, delimiter=',')

def trainingz(data, labels, targets, weights, eta, epoch):
    for current_epoch in range(0, epoch):
        accuracy = 0
        for i in range(0, len(data)):
            tk = labels[i]
            activation = np.dot(data[i], weights)
            yk = np.argmax(activation)
            output = np.where(activation > 0, 1, 0)
            if current_epoch == 0:
                if tk == yk:
                    accuracy +=1

            else:
                if tk != yk:
                    delta = (targets[tk] - output).reshape(10,1)
                    #output_k = output.reshape(10,1)
                    #target_k = targets[tk].reshape(10,1)
                    data_k = data[i].reshape(1, 785)
                    delta_w = eta * np.dot(delta, data_k)
                    weights = np.add(weights, delta_w.T)
                else:
                    accuracy +=1

        # print('\n weight and new weight are same: ', np.array_equal(hold_weight, weights))
        print('\n epoch: %s, accuracy %s' % (current_epoch, accuracy/data.shape[0]))
    return accuracy

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
	
	# train_file = "mnist_train.csv"
	# test_file  = "mnist_test.csv"

	train_data_file = 'data/train-images.idx3-ubyte'
	train_label_file = 'data/train-labels.idx1-ubyte'


	train_data, train_label = read_data_normalize_and_add_bias(train_label_file, train_data_file)

	
	rows, cols = train_data.shape
	# print('Training\nDimensions: %s x %s' % (rows, cols))
	# print('\n1st row', train_data[0])

	
	# we need 10 sets of weigts, one for each digit we want to id
	weights = np.random.default_rng().uniform(-0.05, 0.05, [cols, 10])
	for i in range(cols):
		weights[i] = [-0.008273979447639121, -0.018066262830690627, -0.017009612515183792, -0.026791061367860403, 0.01926989680582465, -0.011686243205535941, 0.0015042996616475152, 0.048184713508116275, -0.04348513962852357, 0.022143747122333163]

	# when identifies digit 0 we get the zeroth row and one-hot matrix
	targets = np.identity(10, dtype=int)
	
	# targets = np.identity(10, dtype=float)
	# alternative
	# targets = one_hot_encoded_Targets(train_label, rows)
	# targets = np.zeros([rows, 10])
	# targets[train_label] = 1;

	#adjust weights to match target
	# train(epochs, eta, train_data, train_label, weights, targets, batch)
	# tup = training(epochs, eta, train_data, train_label, weights, targets, batch)

	tup = trainings(epochs, eta, train_data, train_label, weights, targets, batch)
	# tup = trainingz(train_data, train_label, targets, weights, eta, epochs)


	train_accuracy_list = tup[0]
	final_weights = tup[1]
	trained_epochs = tup[2]
	all_trained_Weight_sets = tup[3]
	epochs = len(trained_epochs)


	test_data_file   = 'data/t10k-images.idx3-ubyte'
	test_labels_file = 'data/t10k-labels.idx1-ubyte'
	
	test_data, test_labels = read_data_normalize_and_add_bias(test_labels_file, test_data_file)
	rows, cols = test_data.shape

	# test_labels = one_hot_encoded_Targets(tt, rows)
		
	print('Dimensions: %s x %s' % (rows, cols))
  	# print('\n1st row', pret_data[0])
	
	# gets correct % prediction for each epoch (list)
	testCorrect = test(epochs, test_data, all_trained_Weight_sets, test_labels, batch)

	# prints accuracy graph for training and test data
	accuracyGraph(train_accuracy_list, trained_epochs , testCorrect)

	# printf confusion matrix for the test data
	confMat(test_data, test_labels, final_weights, epochs, batch, eta)



if __name__ == "__main__":
    main()




	# if __name__ == "__main__":
    # targets = np.identity(10, dtype=float)
    # # weights = np.random.uniform(-.05, .05, (data.shape[1], targets.shape[0]))
    # train_data, train_labels = load_normalize('/home/zeilo/Documents/machine_learning/hw_1/train-images.idx3-ubyte','/home/zeilo/Documents/machine_learning/hw_1/train-labels.idx1-ubyte')
    # weights = np.array(np.random.uniform(-.05, .05, (train_data.shape[1], targets.shape[0])), dtype='f')
    # eta = .1
    # epoch = 10
    # accuracy = training(train_data, train_labels, targets, weights, eta, epoch)