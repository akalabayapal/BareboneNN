from PIL import Image
import numpy as np

def load_image_to_array(image_path:str,size_y:int,size_x:int)->np.array:

    '''
    Resizes the image to the given size and then loads the image into a Numpy array.
    '''

    img = Image.open(image_path)

    img_resized = img.resize((size_y, size_x))  # Standard input size
    return np.asarray(img_resized)

def load_images_to_array(size_y:int,size_x:int,channels:int,image_paths):

    # allocate the array first
    img_array = np.ndarray(
        shape=(len(image_paths),size_y,size_x,channels)
    )

    for index,path in enumerate(image_paths):
        # load and resize the single image
        img = Image.open(path)
        img_array[index] = np.asarray(img.resize((size_y,size_x)))

    return img_array

def im2col(img_matrix: np.ndarray, kernel: np.ndarray,bias:np.ndarray,stride:int=1):
    '''
    Memory-optimized im2col convolution
    '''

    num_filters = kernel.shape[0]
    res_y = kernel.shape[1]
    res_x = kernel.shape[2]

    N, length_row, length_col, num_channels = img_matrix.shape

    out_h = (length_row - res_y) // stride + 1
    out_w = (length_col - res_x) // stride + 1

    # 2. Extract sliding spatial windows across Height (axis 1) and Width (axis 2)
    # Shape of windows: (N, out_h, out_w, num_channels, res_y, res_x)
    windows = np.lib.stride_tricks.sliding_window_view(
        img_matrix, 
        window_shape=(res_y, res_x), 
        axis=(1, 2)
    )

    if stride != 1:
        windows = windows[:,::stride,::stride]


    # 3. Reshape patches into a 2D matrix: (N * out_h * out_w, patch_len)
    big_array = windows.reshape(-1, num_channels * res_y * res_x)

    # 4. Flatten kernels to match patch order: (patch_len, num_filters)
    # Natural C-order matching replaces the need for order='F'
    kernel_flat = kernel.astype(np.float32).reshape(num_filters, -1).T


    # 5. Zero-copy matrix multiplication across ALL images and ALL filters simultaneously
    ret = big_array @ kernel_flat  + bias # Shape: (N * out_h * out_w, num_filters)

    # 6. Reshape back into 4D Tensor output: (N, out_h, out_w, num_filters)
    feature_maps = ret.reshape(N, out_h, out_w, num_filters)

    return feature_maps,big_array

def col2im(dX_col, input_shape, kernel_size, stride=1):
    """
    Accumulates patch gradients back into full 4D input shape (N, H, W, C).
    """
    N, H, W, C = input_shape
    k_h, k_w = kernel_size
    
    out_h = (H - k_h) // stride + 1
    out_w = (W - k_w) // stride + 1

    # 1. Initialize output gradient tensor with zeros
    dX = np.zeros(input_shape, dtype=np.float32)

    # 2. Reshape patch gradients: (N, out_h, out_w, k_h, k_w, C)
    dX_col_reshaped = dX_col.reshape(N, out_h, out_w, k_h, k_w, C)

    # 3. Accumulate (+) overlapping patch gradients back into dX
    for r in range(k_h):
        for c in range(k_w):
            dX[:, r:r + out_h * stride:stride, c:c + out_w * stride:stride, :] += dX_col_reshaped[:, :, :, r, c, :]

    return dX

def handle_maxpooling(image_tensor:np.array,mask_y:int,mask_x:int,stride:int):

    # make the grid first 
    '''
    This makes the blocks of (mask_y x mask_x) only along rows and cols of the image. 
    Other layers (N and C) remains intact
    '''
    windows = np.lib.stride_tricks.sliding_window_view(
            image_tensor, 
            window_shape=(mask_y, mask_x), 
            axis=(1, 2)
        )

    strided_window:np.ndarray = windows[:,::stride,::stride] # apply the given stride to the image

    # calculate the max in each sub grids made by the sliding_window_view
    maxpooled = np.max(strided_window,axis=(-2,-1))

    # get the new shape
    shape_new = list(strided_window.shape[:-2])
    shape_new.append(mask_x * mask_y)

    max_windows = np.argmax(
        strided_window.reshape(tuple(shape_new)),
        axis=-1
    )

    return maxpooled,max_windows

def handle_maxpool_bp_tensor(original_image_tensor_shape:tuple,mask_pos_arr:np.ndarray,error_tensor:np.ndarray,mask_y:int,mask_x:int,stride:int):
    '''
    Makes the error tensor and return it.
    '''

    out_h, out_w = error_tensor.shape[1], error_tensor.shape[2]

    # this will contain the position of the argmax
    shape_mask_pos = list(mask_pos_arr.shape)
    shape_mask_pos.append(mask_x * mask_y)

    shape_decomposed = list(mask_pos_arr.shape)
    shape_decomposed.extend([mask_y,mask_x])

    # 4. Expand 4D arrays to 5D to match tensor_composed shape
    mask_pos_expanded = np.expand_dims(mask_pos_arr, axis=-1)
    error_tensor_expanded = np.expand_dims(error_tensor, axis=-1)

    # put the errors at correct positions using put_along axis
    tensor_composed = np.zeros(shape=tuple(shape_mask_pos),dtype=np.float32)
    np.put_along_axis(tensor_composed,mask_pos_expanded,error_tensor_expanded,-1)

    # deompose the tensor into the mask_y,mask_x components
    tensor_decomposed = tensor_composed.reshape(tuple(shape_decomposed))

    # this is the tensor to return with 0 in non-max places and total accumulated error in the places of error
    dX = np.zeros(shape=original_image_tensor_shape,dtype=np.float32)

    for r in range(mask_y):
        for c in range(mask_x):
            dX[:, r:r + out_h * stride:stride, c:c + out_w * stride:stride, :] += tensor_decomposed[:, :, :,:, r, c]


    return dX
    
def handle_avgpooling(image_tensor:np.array,mask_y:int,mask_x:int,stride:int):

    # make the grid first 
    '''
    This makes the blocks of (mask_y x mask_x) only along rows and cols of the image. 
    Other layers (N and C) remains intact
    '''
    windows = np.lib.stride_tricks.sliding_window_view(
            image_tensor, 
            window_shape=(mask_y, mask_x), 
            axis=(1, 2)
        )

    strided_window:np.ndarray = windows[:,::stride,::stride] # apply the given stride to the image

    # calculate the max in each sub grids made by the sliding_window_view
    avgpooled = np.mean(strided_window,axis=(-2,-1))

    return avgpooled

def handle_avgpool_bp_tensor(original_image_tensor_shape:tuple,error_tensor:np.ndarray,mask_y:int,mask_x:int,stride:int):

    size_kernel = mask_y * mask_x # store the total kernel size for taking mean later
    N,H,W,C = original_image_tensor_shape

    out_h, out_w = error_tensor.shape[1], error_tensor.shape[2]

    err_scaled = error_tensor / size_kernel

    # expand dimensions
    expanded_err = np.expand_dims(err_scaled,axis=(-2,-1))

    # use boradcast_to to copy the errors to other dims also . Basically convert 4D tensor to 6D
    tensor_decomposed = np.broadcast_to(expanded_err,(N,out_h,out_w,C,mask_y,mask_x))


    # this is the tensor to return with 0 in non-max places and total accumulated error in the places of error
    dX = np.zeros(shape=original_image_tensor_shape,dtype=np.float32)

    for r in range(mask_y):
        for c in range(mask_x):
            dX[:, r:r + out_h * stride:stride, c:c + out_w * stride:stride, :] += tensor_decomposed[:, :, :,:, r, c]


    return dX

    

