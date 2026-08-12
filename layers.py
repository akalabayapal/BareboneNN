# This contains all activation functions for the neural network

import numpy as np


class LinearLayer:

    def __init__(self,neurons_number:int,learning_rate:float,init_method:str=None):
        # m : number of neurons
        # n : number of features
        self.nos_nerons = neurons_number

        self.init_method = init_method
        self.weights = None # this will be initalized later (m x n)
        self.data = None # this will contain the data matrix  (k x n)

        self.forward_data = None # (k x m)
        self.lr_rate = learning_rate

        self.b = np.zeros((1, neurons_number))

        


        self.error_normal = None # (k x n)
        self.error_b = None

    def initalize(self,n_in,next_layer):

        if self.init_method == None:

            if isinstance(next_layer,ReluLayer) or isinstance(next_layer,EluLayer) or isinstance(next_layer,LeakyReluLayer):
                self.init_method = 'he'
            elif isinstance(next_layer,SigmoidLayer) or isinstance(next_layer,TanhLayer):
                self.init_method = 'xavier'

        # Select weight initialization strategy
        if self.init_method == "he":
            limit = np.sqrt(2.0 / n_in)
            self.weights = np.random.randn(n_in, self.nos_nerons) * limit
        elif self.init_method == "xavier":
            limit = np.sqrt(2.0 / (n_in + self.nos_nerons))
            self.weights = np.random.randn(n_in, self.nos_nerons) * limit
        else:
            # Simple small random values
            self.weights = np.random.randn(n_in, self.nos_nerons) * 0.01


    def forward(self):

        # this will calculate the y = w1x1 + w2x2 + w3x3 ....
    
        self.forward_data = self.data @ self.weights +self.b
        return self.forward_data

    def inference(self,data:np.array):

        return data @ self.weights + self.b

    def backward(self, error: np.ndarray):
        # 1. Compute weight gradients (X^T @ error)
        grad_weights = self.data.T @ error

        # 2. Compute bias gradients
        self.error_b = np.sum(error, axis=0, keepdims=True) 

        # 3. Compute error to pass back to previous layer (error @ W^T)
        self.error_normal = error @ self.weights.T  

        # 4. Update weights using learning rate
        self.weights -= self.lr_rate * grad_weights
        self.b -= self.lr_rate * self.error_b   

        return self.error_normal

class SigmoidLayer:

    def __init__(self):

        # m : number of features
        self.data = None # (k x m)

        self.forward_data = None # (k x m)
        self.lr_rate = None

        self.error = None # (k x m)

    def forward(self):

        # take the sigmoid

        self.forward_data = 1 / (1 + np.exp(-1 * self.data))

        return self.forward_data

    def inference(self,data:np.array):

        return 1 / (1 + np.exp(-1 * data))

    def backward(self,error:np.array):

        self.error = error * (self.forward_data*(1 - self.forward_data))
        return self.error

class ReluLayer:

    def __init__(self):

        # m : number of features

        self.data = None # this will contain the data matrix  (k x m)

        self.forward_data = None # (k x m)
    
        self.error = None # (k x m)

    def forward(self):

        self.forward_data = np.maximum(0,self.data)
        return self.forward_data

    def inference(self,data:np.array):

        return np.maximum(0,data)

    def backward(self,error):

        # Create a mask of 1s and 0s: True (1) where data > 0, False (0) otherwise
        relu_grad = (self.data > 0).astype(float)

        # Element-wise multiplication with incoming error
        self.error = error * relu_grad
        return self.error

class LeakyReluLayer:

    def __init__(self,alpha):

        # m : number of features
        
        self.data = None # this will contain the data matrix  (k x m)
        self.forward_data = None # (k x m)
        self.error = None # (k x m)
        self.alpha = alpha

    def forward(self):

        self.forward_data = np.where(self.data >= 0,self.data,self.data*self.alpha)
        return self.forward_data

    def inference(self,data:np.array):

        return np.where(data >= 0,data,data * self.alpha)


    def backward(self,error):

        self.error = error * np.where(
            self.data >= 0,
            1.0,
            self.alpha

        )
        return self.error
 
class EluLayer:

    def __init__(self,alpha:float):
       # m : number of features

        self.data = None # this will contain the data matrix  (k x m)

        self.forward_data = None # (k x m)
        self.alpha = alpha
    
        self.error = None # (k x m)

    def forward(self):
  
        # f(x) = x if x > 0 else alpha * (exp(x) - 1)
        self.forward_data = np.where(self.data > 0, self.data, self.alpha * (np.exp(self.data) - 1))
        return self.forward_data

    def inference(self,data):
        return np.where(data > 0 ,data,self.alpha * (np.exp(data) - 1))

    def backward(self, error):
        # f'(x) = 1 if x > 0 else alpha * exp(x)
        elu_grad = np.where(self.data > 0, 1.0, self.alpha * np.exp(self.data))
        self.error = error * elu_grad
        return self.error

class TanhLayer:

    def __init__(self):
       # m : number of features

        self.data = None # this will contain the data matrix  (k x m)

        self.forward_data = None # (k x m)
    
        self.error = None # (k x m)

    def forward(self):

        self.forward_data = np.tanh(self.data)
        return self.forward_data

    def inference(self,data):

        return np.tanh(data)

    def backward(self,error):

        self.error = error * (1 - self.forward_data ** 2)
        return self.error

class SoftmaxLayer:

    def __init__(self,eps=1e-15):
        

       # m : number of features

        self.data = None # this will contain the data matrix  (k x m)   
        self.forward_data = None # (k x m)
        self.error = None # (k x m)

        self.sum = None
        self.eps = eps

        self.y = None

        unique_y = len(np.unique(self.y))
        self.one_hot = np.eye(unique_y,dtype=int)[self.y]

        self.last:bool = False

    def forward(self):



        self.scaled = np.exp(self.data - np.max(self.data,keepdims=True,axis=1))
        self.sum = np.sum(self.scaled,keepdims=True,axis=1)
        self.forward_data = self.scaled / self.sum

        return self.forward_data

    def inference(self,data):

        scaled = np.exp(data - np.max(data,keepdims=True,axis=1))
        s = np.sum(scaled,keepdims=True,axis=1)
        return scaled / s
        



    def backward(self, error: np.ndarray):

        # CrossEntropy MultiClass autohandles the errors on its own no need to change anything
        # Implementing softmax gradient and CrossEntropy gradient separately is mathematically messy hence. Those are implemented toghether

        # if the next layer is not the Loss layer we will need to do the work here itself. Else we will gracefully pass the same error backwards
        # considering the next layer is the Loss layer..

        if self.last:
            return error 

        # we need to calculate the softmax gradient here

        # 2. Middle layer standalone Softmax
        # Assuming self.forward_data is the saved Softmax output P of shape (k, C)
        P = self.forward_data

        # Step A: Compute row-wise sum of (error * P) across classes (axis=-1)
        sum_dP = np.sum(error * P, axis=-1, keepdims=True)  # Shape: (k, 1)

        # Step B: Apply full Jacobian product: P * (error - sum_dP)
        self.error = P * (error - sum_dP)  # Shape: (k, C)

        return self.error

        
            
                   




    
        














        











    

    

    
        

    

    

    

    


    

    

