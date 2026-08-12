# this file contains implementation of loss functions

import numpy as np 

class MSELoss:
    def __init__(self,Y:np.array):

        self.Y = Y
        self.error = []
        self.grad_err = None


    def get_errors(self,last_data:np.array):
        '''
        Computes the errors and gives back to prev layer
        '''

        if last_data.shape[1] != 1:
            raise TypeError(f'The output layer must have a output of shape (j,1) which do not match the shape:({last_data.shape[0]},{last_data.shape[1]})')

        # convert it into row major


        self.grad_err = np.array([last_data[:,0] - self.Y]).T 

        self.error.append(np.mean(self.grad_err ** 2))

        return self.grad_err


class MAELoss:
    def __init__(self,Y:np.array):
    
            self.Y = Y
            self.error = []
            self.grad_err = None

    def get_errors(self,last_data:np.array):
            
        '''
        Computes the errors and gives back to prev layer
        '''

        if last_data.shape[1] != 1:
            raise TypeError(f'The output layer must have a output of shape (j,1) which do not match the shape:({last_data.shape[0]},{last_data.shape[1]})')

        # convert it into row major


        err= np.array([last_data[:,0] - self.Y]).T 

        self.grad_err = np.sign(err)/last_data.shape[0]

        self.error.append(np.mean(np.abs(err)))

        return self.grad_err
    


class CrossEntropyBinaryLoss:
    def __init__(self, Y: np.ndarray, eps: float = 1e-15):
        # Reshape Y to column vector (k, 1) to match last_data shape guaranteed
        self.Y = Y.reshape(-1, 1)
        self.eps = eps
        self.error = []
        self.grad_err = None

    def get_errors(self, last_data: np.ndarray) -> np.ndarray:
        '''
        Computes the BCE scalar loss and returns the gradient to the previous layer.
        Expects last_data shape: (k, 1) and Y shape: (k, 1)
        '''
        if last_data.shape[1] != 1:
            raise TypeError(
                f'The output layer must have an output of shape (k, 1), '
                f'got ({last_data.shape[0]}, {last_data.shape[1]}) instead.'
            )

        # 1. Clip predictions to prevent log(0) and division by zero
        preds = np.clip(last_data, self.eps, 1.0 - self.eps)

        # 2. Scalar Loss Calculation: - [y * log(p) + (1 - y) * log(1 - p)]
        loss_matrix = -(self.Y * np.log(preds) + (1.0 - self.Y) * np.log(1.0 - preds))
        self.error.append(np.mean(loss_matrix))

        k = last_data.shape[0]

        self.grad_err = (preds - self.Y) / (preds * (1 - preds) * k)

        return self.grad_err

class CrossEntropyMultiClassLoss:
    def __init__(self,Y:np.array,eps=1e-15):
        
        self.Y = Y
        self.error = []
        self.grad_err = None
        self.eps = eps

        self.unique_y = len(np.unique(self.Y))

        self.one_hot = np.eye(self.unique_y,dtype=int)[self.Y]


    def get_errors(self,last_data:np.ndarray) -> np.ndarray:
        '''
        Computes the MCE scalar loss and returns the gradient to the previous layer.
        Expects last_data shape: (k, n) and Y shape: (k, 1)
        Where n are number of classes of the data
        '''
        if last_data.shape[1] != self.unique_y:
            raise TypeError(
                f'The output layer must have an output of shape (k, n), Where n=Number of unique classes in dataset '
                f'got ({last_data.shape[0]}, {last_data.shape[1]}) instead.'
            )

        self.grad_err = (last_data - self.one_hot)/last_data.shape[0]

        # 1. Clip predictions to avoid log(0)
        clipped_preds = np.clip(last_data, self.eps, 1 - self.eps)

        # 2. Element-wise multiplication and mean loss
        self.error.append(-np.mean(np.sum(self.one_hot * np.log(clipped_preds), axis=1)))

        return self.grad_err

        

        

