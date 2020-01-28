
# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np

class pcn:
	""" A basic Perceptron"""
	
	def __init__(self,logical_OR,logical_OR_targets):
		""" Constructor """
		# Set up network size
		if np.ndim(logical_OR)>1:
			self.nIn = np.shape(logical_OR)[1]
		else: 
			self.nIn = 1
	
		if np.ndim(logical_OR_targets)>1:
			self.nOut = np.shape(logical_OR_targets)[1]
		else:
			self.nOut = 1

		self.nData = np.shape(logical_OR)[0]
	
		# Initialise network
		self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.02
		# self.weights = np.empty([self.nIn+1, self.nOut])
		# self.weights[0] = -0.05
		# self.weights[1] = -0.02
		# self.weights[2] = 0.02


	def pcntrain(self,inputs,targets,eta,nIterations):
		""" Train the thing """	
		# Add the inputs that match the bias node
		bais = -np.ones([self.nData, 1], dtype=int)
		# axis = 1 means vertical,   axis = 0 means horizontal
		inputs = np.concatenate((bais, inputs), axis=1)
		# Training
		change = range(self.nData)

		for n in range(nIterations):
			print("Iteration: %d"% n)
			print(self.weights)
			self.activations = self.pcnfwd(inputs, targets)
			print("Inputs: ")
			print(inputs)
			print("Final outputs are: ")
			print(self.activations)
		
			# Randomise order of inputs
			#np.random.shuffle(change)
			#inputs = inputs[change,:]
			#inputs_targets = inputs_targets[change,:]
			
		#return self.weights

	def pcnfwd(self, inputs, targets):
		""" Run the network forward """
		activations = np.zeros([self.nData, 1], dtype=int)
		# Compute activations
		for k in range(self.nData):
			# activations =  np.dot(logical_OR[i],self.weights)
			sotty = np.dot(inputs[k],self.weights)
			activations[k] = 1 if  sotty > 0 else 0
			if activations[k] != targets[k]:
				# yt = np.empty([self.nData,1])
				yt = np.repeat(activations[k] - targets[k], self.nData)
				iT = np.transpose(inputs[k])
				for i in range(self.nIn+1):
					yt = activations[k][0] - targets[k][0]
					self.weights[i][0] -= eta * (yt) * inputs[k][i]
				print("New Weights")
				print(self.weights)

			

		# Threshold the activations
		return np.where(activations>0,1,0)


	def confmat(self,logical_OR,logical_OR_targets):
		"""Confusion matrix"""

		# Add the logical_OR that match the bias node
		logical_OR = np.concatenate((logical_OR,-np.ones((self.nData,1))),axis=1)
		
		outputs = np.dot(logical_OR,self.weights)
	
		nClasses = np.shape(logical_OR_targets)[1]

		if nClasses==1:
			nClasses = 2
			outputs = np.where(outputs>0,1,0)
		else:
			# 1-of-N encoding
			outputs = np.argmax(outputs,1)
			logical_OR_targets = np.argmax(logical_OR_targets,1)

		cm = np.zeros((nClasses,nClasses))
		for i in range(nClasses):
			for j in range(nClasses):
				cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(logical_OR_targets==j,1,0))

		print(cm)
		print(np.trace(cm)/np.sum(cm))
		
def logic():
	import pcn
	""" Run AND and XOR logic functions"""

	logical_AND = np.array([[1,1,1],
							[1,0,0],
				  			[0,1,0],
				  			[0,0,0]])
	logical_XOR = np.array([[1,1,0],
				  			[1,0,1],
				  			[0,1,1],
					   		[0,0,0]])
	logical_NOT = np.array([[1,0],
							[0,1]])

	p = pcn.pcn(logical_AND[:,0:2],logical_AND[:,2:])
	p.pcntrain(logical_AND[:,0:2],logical_AND[:,2:],0.25,10)
	p.confmat(logical_AND[:,0:2],logical_AND[:,2:])

	q = pcn.pcn(logical_XOR[:,0:2],logical_XOR[:,2:])
	q.pcntrain(logical_XOR[:,0:2],logical_XOR[:,2:],0.25,10)
	q.confmat(logical_XOR[:,0:2],logical_XOR[:,2:])


logical_OR = np.array([ [0,0],
					[0,1],
					[1,0],
					[1,1]])
logical_OR_targets = np.array([[0],
					[1],
					[1],
					[1]])

logical_NOT = np.array([ [1],
						 [0]])
logical_NOT_targets = np.array([[0],
								[1]])

eta = 0.25
T = 6
# print("Created Perceptron")
# p = pcn(logical_OR, logical_OR_targets)
# print("Starting the training")
# p.pcntrain(logical_OR, logical_OR_targets, eta, T)		
print("NOT")		
print("Created Perceptron")
p = pcn(logical_NOT, logical_NOT_targets)
print("Starting the training")
p.pcntrain(logical_NOT, logical_NOT_targets, eta, T)	
print("NOT")		