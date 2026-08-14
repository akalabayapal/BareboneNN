import numpy as np
from layers import *
import gc

class BareboneNN:

    def __init__(self,*layers):

        self.layers_list = layers
        self.loss_function = None

    def fit(self,X:np.ndarray,Y:np.ndarray,epoch:int,batch_size:int=None):

        '''
        X: Numpy array of data matrix
        Y: Labels or prediction data
        epoch: Number of times to rotate the data for training
        batch_size: The number of data to rotate at a single time (DEFAULT:Whole data is processed at once)

        WARNING: While training ensure to add a considerable batch size.
        '''
        if batch_size == None:
            batch_size = X.shape[0] # If no batch size is specified then whole content is loaded at once

        self.X = X
        self.Y = Y
        self.batch_size = batch_size # store the batch size


        for layer_index,layer in enumerate(self.layers_list):
        
            if isinstance(layer,LinearLayer):
                if layer_index + 1 == len(self.layers_list):
                    next_layer = None
                else:
                    next_layer = self.layers_list[layer_index+1]
                layer.initalize(next_layer=next_layer)
        
            elif isinstance(layer,SoftmaxLayer):
                layer.y = self.Y
                # check if this is the last layer
                if layer_index == len(self.layers_list) - 1:
                    layer.last = True # yes, this is the last layer
            elif hasattr(layer, "initalize") and callable(getattr(layer,"initalize")):
                layer.initalize() # initalze the function if the weigh are needed to be initalized before hand
        
        

        self.__train(epoch)

    
    def __train(self,epoch:int):
        if self.loss_function == None:
            raise RuntimeError('Please specify a valid Loss function.\n' \
            'use: import loss_function\n' \
            'BareboneNN_instance.loss_function = <Any-loss-function>')

        total = self.X.shape[0]

        # print("batch_size:",self.batch_size)

        for e in range(epoch):
            for batch in range(0,total,self.batch_size):
                self.__train_batch(self.X[batch:batch+self.batch_size],self.Y[batch:batch+self.batch_size])
                # print(f"Training Epoch ({e+1}/{epoch}):Batch {batch+1} completed....")

            gc.collect()


    def model_eval(self):

        '''
        This cleans up all cached intermediate matrices, and data during training. This should be ran before dumping to storage or else model size might be huge.
        '''
        # cleaning up excess data
        del self.X
        del self.Y

        for layer in self.layers_list:

            if hasattr(layer,'forward_data'):
                del layer.forward_data
            if hasattr(layer,'data'):
                del layer.data

        gc.collect()


    def __train_batch(self,batch_x,batch_y):
        # handler to train a single batch
        
        # forward movement
        self.layers_list[0].data = batch_x
        forward_data = self.layers_list[0].forward()

        for layer in self.layers_list[1:]:
            layer.data = forward_data
            forward_data = layer.forward()

        # backward movement
        error = self.loss_function.get_errors(forward_data,batch_y)

        for layer in self.layers_list[::-1]:
            error = layer.backward(error)

        return error

    def predict(self,X):

        forward_data = self.layers_list[0].inference(X)
        
        for layer in self.layers_list[1:]:
            forward_data = layer.inference(forward_data)

        return forward_data

    @property
    def errors(self):
        return self.m.error

    @property
    def error(self):
        return self.m.error[-1]

        



