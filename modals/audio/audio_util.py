import soundfile as sf
import numpy as np

def read_sound_as_array(file_path:str)->np.ndarray:
    '''
    Reads the sound as a numpy array of shape (k,C)
    k = The number of audio samples
    C = Number of streos
    '''

    audio_stereo, sample_rate = sf.read(file_path)
    if len(audio_stereo.shape) == 1:
        return np.expand_dims(audio_stereo,-1)
    return audio_stereo


def read_sounds_as_array(file_paths)->np.ndarray:

    '''
    Reads a list of sounds and returns a array of shape (N,k,C)
    N = Number of audio
    k = The number of audio samples taken
    C = Number of steros

    '''

    arr = []

    for path in file_paths:
        arr.append(read_sound_as_array(path))
    return np.array(arr)