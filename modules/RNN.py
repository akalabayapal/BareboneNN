'''
Implementation of RNN (Recurrent Neural Network) 

We need to implement two important layers to use legacy bareBoneNN internally

Layers needed:

    1. RnnInLayer() => This is the layer goes when data enters the RNN block.
        - It is basically a Linear/Dense Layer that has weights of shape (n, k * 2)
        k1 = For this layer
        k2 = For layer h-1
        N = Number of neurons_in

        - Inputs from previous state is (N,k) and this state (N,k) , they are stacked to form
        (N, 2 * k) and this is used to output a weight

        - neurons_in : N * 2
        - neurons_out : N

        Abstract: LinearLayer(n_in = neurons_out * 2,n_out=neurons_out,learning_rate=lr)
'''

from layers import LinearLayer
import numpy as np

class RnnInLayer:
    def __init__(self,num_features_in:int,num_hidden_features:int,learning_rate:float):
        '''
        features_in: Number of features to be fed into the hidden layer
        learning_rate: The learning rate of the weights
        '''
        self.abstract_instance = LinearLayer(
            neurons_in= num_features_in + num_hidden_features, # we need to send the whole
            neurons_out= num_hidden_features, # get back the data in the shape of hidden state
            learning_rate=learning_rate # forward the learning rate to the abstract
        )

        self.num_features_in = num_features_in

        self.input_this_layer = None # Input of this layer
        self.input_hidden_prev = None # The input from the previous hidden layer

        self.forward_data = None # Data to forward
        self.error = None # This is the error to be back propagated or slice of it will be back propagated

    def initalize(self):
        # initalze the weights of the LinearLayer
        self.abstract_instance.initalize()



    def forward(self):
        '''
        Use the inner Linearlayer abstract for data forwarding
        '''

        # stack two data from this layer and previous layer in coloumn
        # shape initally of each was (k,n) ( k = number of data row, n = features/neuron data of each data row)
        # shape finnally of each is (k,n+m) (k = number of data row, n = features/neuron from this layer + featues/neuron data of prev layer)
        stacked_data = np.hstack((self.input_this_layer,self.input_hidden_prev))

        # send the data to the linearLayer
        self.abstract_instance.data = stacked_data
        self.forward_data = self.abstract_instance.forward()

        return self.forward_data # shape of (k,n): This is the shape we wanted

    def backward(self,error):

        err_total = self.abstract_instance.backward(error=error) # shape: (k,n+m)

        # This is the error to be backpropagated to the Input Layer
        self.error = err_total[:,:self.num_features_in] # shape: (k,n)

        # return the sliced error
        return err_total[:,self.num_features_in:] # shape: (k,m)








