# This file makes a extra layer for CNN with BareboneNN

'''
Custom Layer Structure

forward()
backward(error) the error from the last layer

initalize() # if needed 


forward: must forward the transformed data to next layer eg. return self.forward_data
backward: must return the error for the current layer. So the BareBoneNN's LayerScheduler can backpropagate the error

'''
from modals.CNN.imgutils import *

import numpy as np

class Conv2DLayer:

    def __init__(
            self,
            filters_in:int,
            kernel_y:int,
            kernel_x:int,
            num_channels:int,
            learning_rate:float,
            stride:int=1
            ):

        '''
        Uses optimised im2col method for calculating convolutions and returning the 4D tensor

        FORWARD:
        input = tuple of img_matrix or 4D-img-tensor
            - If the input is a tuple it is converted to a 4D-img-tensor (N,H,W,C)
                1. N = total number of images
                2. H = Height of image(in pixel count)
                3. W = Height of image(in pixel count)
                4. C = Number of channels

        output = a 4D-img-tensor of (N,out_y,out_x,C)
            1. out_y,out_x: The new resolution after batching
        '''

        self.filters_num = filters_in
        self.kernel_y = kernel_y
        self.kernel_x = kernel_x
        self.num_channels = num_channels

        self.lr = learning_rate
        self.stride = stride

        self.filters = None
        self.bias = None

        self.data = None

        self.forward_data = None
        self.error = None


    def initalize(self):

        # 1. Define filter hyper-parameters
        num_filters = self.filters_num      # Number of output feature maps (F)
        res_y, res_x = self.kernel_y,self.kernel_x  # Kernel height and width
        in_channels = self.num_channels      # Input channels (e.g., RGB)

        # 2. Calculate "fan_in" (total weights per filter)
        fan_in = res_y * res_x * in_channels

        # 3. He (Kaiming) Normal Initialization
        # Scale factor: sqrt(2 / fan_in)
        scale = np.sqrt(2.0 / fan_in)
        self.filters = np.random.randn(num_filters, res_y, res_x, in_channels) * scale
        self.bias = np.zeros((1,num_filters)) 


    def forward(self):

        # if isinstance(self.data,tuple):
        #     self.data = np.stack(self.data)
        #     self.forward_data,self.big_array = im2col(self.data,self.filters,self.bias,stride=self.stride)
        # else:
        self.forward_data,self.big_array = im2col(self.data,self.filters,self.bias,stride=self.stride)
        return self.forward_data

    def inference(self,data):

        # this is for prediction
        tf = im2col(data,self.filters,self.bias,stride=self.stride)[0]
        return tf
    
        

    def backward(self,error_4d:np.ndarray):

        N, H, W, C = self.data.shape  # Cached from forward pass
        
        # 1. Reshape 4D error into 2D matrix
        dY_2d = error_4d.reshape(-1, self.filters_num)

        # Shape big-array: (N * h * w ,res_y*res_X*channels) 
        #dy_2d = (h * w * N,number_of_channels)
        # dw_flat = (res_y,res_x*channels,number_of_chanels)
        dW_flat = self.big_array.T @ dY_2d

        # shape = (filters_num,res_y,res_x,channels)
        self.filters -= dW_flat.T.reshape(self.filters_num,self.kernel_y,self.kernel_x,self.num_channels) * self.lr

        # Fixing the bias properly
        # the total accumulated error of each bias is the sum off errors ofeach batch(in row).
        # so we take the sum of all errors along the row axis=0
        self.bias -= np.sum(dY_2d,axis=0,keepdims=True) * self.lr

        # Now we need to calculate the error dX so that we can pass the error to layers behind
        dX = (dY_2d @ dW_flat.T)



        e_new =  col2im(
            dX,self.data.shape,(self.kernel_y,self.kernel_x)
        )

        # Cleaning of the data right now as this batch processing is over 
        del self.forward_data
        del self.big_array
        del self.data

        return e_new
  
class FlattenLayer:
    '''
    Flattens the 4D image tensor to a 2D matrix that can be processed by conventional NN
    (N,out_y,out_x,channels) --> (N,out_y*out_x*channels)
    '''
    def __init__(self):
        self.data = None
        self.forward_data = None
        self.error = None

    def forward(self):
        self.org_shape = self.data.shape
        self.forward_data = self.data.reshape(self.org_shape[0],-1)

        return self.forward_data

    def inference(self,data):
        rs =  data.reshape(data.shape[0],-1)
        return rs


    def backward(self,error:np.ndarray):

        self.error = error.reshape(self.org_shape)
        return self.error

class MaxPool2D:
    def __init__(self,mask_y:int,mask_x:int,stride:int=1):

        self.data = None
        self.forward_data = None

        # save the mask sizes
        self.mask_y = mask_y
        self.mask_x = mask_x
        self.stride = stride

    def forward(self):

        # Calculates the maxpool sampled data and stores the argmax(positions) to fix the gradients
        self.forward_data,self.max_pos = handle_maxpooling(self.data,self.mask_y,self.mask_x,self.stride)

        return self.forward_data

    def inference(self,data):
        return handle_maxpooling(data,self.mask_y,self.mask_x,self.stride)[0]


    def backward(self,error):
        return handle_maxpool_bp_tensor(self.data.shape,self.max_pos,error,self.mask_y,self.mask_x,self.stride)

class AvgPool2D:
    def __init__(self,mask_y:int,mask_x:int,stride:int=1):
    
        self.data = None
        self.forward_data = None

        # save the mask sizes
        self.mask_y = mask_y
        self.mask_x = mask_x
        self.stride = stride
    
    def forward(self):

        # Calculates the maxpool sampled data and stores the argmax(positions) to fix the gradients
        self.forward_data = handle_avgpooling(self.data,self.mask_y,self.mask_x,self.stride)

        return self.forward_data

    def inference(self,data):

        return handle_avgpooling(data,self.mask_y,self.mask_x,self.stride)

    def backward(self,error):
        return handle_avgpool_bp_tensor(self.data.shape,error,self.mask_y,self.mask_x,self.stride)
    


