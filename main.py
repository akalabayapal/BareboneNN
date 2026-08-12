# Main implementation of bareboneNN

from layers import *
from loss_function import CrossEntropyMultiClassLoss

class BareboneNN:

    def __init__(self,*layers):

        # process the layers one by one and add it to a list
        if not isinstance(layers[0],LinearLayer):
            raise TypeError('The input layer must be dense(LinearLayer)')

        self.layers_list = layers
        self.loss_function = None

    

    def fit(self,X,Y,epoch:int):

        input_layer:LinearLayer = self.layers_list[0]
        input_layer.data = X

        self.Y = Y

        # initalize the linear layers
        n_in = X.shape[1] # gets number of cols

        for layer_index,layer in enumerate(self.layers_list):

            if isinstance(layer,LinearLayer):
                if layer_index + 1 == len(self.layers_list):
                    next_layer = None
                else:
                    next_layer = self.layers_list[layer_index+1]
                layer.initalize(n_in=n_in,next_layer=next_layer)
                n_in = layer.nos_nerons

            elif isinstance(layer,SoftmaxLayer):
                if layer_index - 1 < 0:
                    raise TypeError('Fist layer can not be a softmax layer.Use a dense layer.')

                if n_in != len(np.unique(self.Y)):
                    raise TypeError('The previous dense layer output neurons must match the number of classes of the data.')

                layer.y = self.Y

                # check if this is the last layer
                if layer_index == len(self.layers_list) - 1:
                    layer.last = True # yes, this is the last layer
                    
                    

        self.__train(epoch)

        


    def __train(self,epoch:int):

        if self.loss_function == None:
            raise RuntimeError('Please specify a valid Loss function.\n' \
            'use: import loss_function\n' \
            'BareboneNN_instance.loss_function = <Any-loss-function>')

        self.m = self.loss_function(self.Y)

        for _ in range(epoch):

            # forward movement

            forward_data = self.layers_list[0].forward()

            for layer in self.layers_list[1:]:

                layer.data = forward_data
                forward_data = layer.forward()


            # backward movement

            error = self.m.get_errors(forward_data)

            for layer in self.layers_list[::-1]:
                error = layer.backward(error)

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
        

        



