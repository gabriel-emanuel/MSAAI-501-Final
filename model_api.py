import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.keras as keras

def NN_classifier(hidden_neurons, hidden_layers, lr=3e-3, gamma=1e-5, verbose=False):
	'''
		hidden_neurons: the number of neurons per hidden layer (int)
		hidden_layers: the number of hidden layers in the MLP (int)
		lr: the learning rate.. it is the coefficient on the gradient descent algorithm (float)
		gamma: the l2 normalization coefficient, it affects the strength of the l2 penalty on loss. (float)
		verbose: if true, print model summary() (boolean)
	'''
	X = keras.layers.Input(shape=(28,28)) #Take a flat vector 28*28
	flatten = keras.layers.Flatten()(X)
	
	#Create an initial hidden layer
	#Relu is the standard activation function, we will use it for all but the final layer.
	
	hidden = keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(gamma))(flatten) #Use Default Initialization
	for h in range(hidden_layers - 1):
		hidden = keras.layers.Dense(hidden_neurons, activation="relu", kernel_regularizer=keras.regularizers.l2(gamma))(hidden)

	# Classification Output
	classifier = keras.layers.Dense(10, activation="softmax", kernel_regularizer=keras.regularizers.l2(gamma))(hidden)

	# Create model and optimizer now!
	model = keras.models.Model(inputs=X, outputs=classifier)

	# Lets use the standard Stochastic Gradient Descent method
	optimizer = keras.optimizers.SGD(learning_rate=lr)

	# For our model we will use categorical crossentropy for classification loss
	# sparse in this context is just the format we choose to have our labels in (a single number denoting the category)
	model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	if verbose:
		model.summary()

	return model

def CNN_classifier(filters, hidden_layers, lr=3e-3, gamma=1e-5, verbose=False):
	'''
		filters: the number of 3x3 filters per CNN layer (int)
		hidden_layers: the number of hidden layers in the MLP (int)
		lr: the learning rate.. it is the coefficient on the gradient descent algorithm (float)
		gamma: the l2 normalization coefficient, it affects the strength of the l2 penalty on loss. (float)
		verbose: if true, print model summary() (boolean)
	'''
	X = keras.layers.Input(shape=(28,28, 1)) #Take an image vector (greyscale)

	#Relu is the standard activation function, we will use it for all but the final layer.

	#Create an initial hidden convolutional layer
	conv = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', kernel_regularizer=keras.regularizers.l2(gamma))(X)

	#No padding, so conv should be (26, 26, 32)
	for h in range(hidden_layers - 1):
		conv = keras.layers.Conv2D(filters, kernel_size=(3,3), activation='relu', kernel_regularizer=keras.regularizers.l2(gamma))(conv)

	max_pool = keras.layers.MaxPooling2D(pool_size=(2,2))(conv)
	# Boureau, Y.L.; Ponce, J.; LeCun, Y. A theoretical analysis of feature pooling in visual recognition. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), Haifa, Israel, 21–24 June 2010; pp. 111–118. [Google Scholar]

	latent = keras.layers.Flatten()(max_pool) # Maybe we can study this latent space vector generated only by the Convolutional Layers?

	# Classification Output
	classifier = keras.layers.Dense(10, activation="softmax", kernel_regularizer=keras.regularizers.l2(gamma))(latent)

	# Create model and optimizer now!
	model = keras.models.Model(inputs=X, outputs=classifier)

	# Lets use the standard Stochastic Gradient Descent method
	optimizer = keras.optimizers.SGD(learning_rate=lr)

	# For our model we will use categorical crossentropy for classification loss
	# sparse in this context is just the format we choose to have our labels in (a single number denoting the category)
	model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	if verbose:
		model.summary()

	return model

# Retrieve Data Set
def get_standard_data():
	(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

	# Normalize Data Set
	x_train = x_train / 255.0
	x_test = x_test / 255.0

	# Use only the training input for normalization.. avoid info leak
	mu = np.mean(x_train)
	std = np.std(x_train)

	x_train = (x_train - mu) / std
	x_test = (x_test - mu) / std

	return x_train, y_train, x_test, y_test