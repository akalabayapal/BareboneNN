# This converts Audio to spectograph histogram then ran into CNN just like image Tensors

import numpy as np

def audio_amp_spec_graph(
        audio_1d_tensor: np.ndarray, # The 1D tensor of the amp of the image
        window_size: int, # The sliding window size of the audio
        bins: int, # number of bins (N) to be used
        strides: int = 1 # the amounts of strides to take
) -> np.ndarray:

    '''
    It takes the Audio 3D (N,Size,Stereos)  tensor to 4D image spectograph 
    X-axis: The time-chunks
    Y-axis: The all bins of audio in that time frames
    '''
    

    # make sliding window of the audio
    windows = np.lib.stride_tricks.sliding_window_view(audio_1d_tensor,window_size,1)

    #strided window of the audio
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))
    strided_window = windows[:,::strides,:] * hann


    N,num_window,channels,window_size = strided_window.shape
   

    data_tensor = (np.array([[num for num in range(window_size)]] * bins).T  * 2 * np.pi) / window_size

    k = np.array([num for num in range(bins)])
   
    scaled_k_tensor = data_tensor * k

    final_tensor = np.cos(scaled_k_tensor) + np.sin(scaled_k_tensor) * -1j


    spec_img = np.transpose(np.abs(strided_window @ final_tensor),axes=(0,3,1,2))

    return spec_img

