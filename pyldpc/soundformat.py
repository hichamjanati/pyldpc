import numpy as np

from .ldpcalgebra import int2bitarray,bitarray2int

__all__=['Audi2Bin','Bin2Audio']

def Audio2Bin(audio_array):
    
    """
    Converts the first audio channel (first column) of an int16 audio_array to a 17-bits binary form.
    
    Parameters:
    - audio-array: must be int16. May be 2D-array but the function only converts one channel. 
    
    returns:
    - 17 bits binary audio-array shaped (length,17) where length is the audio_array's length. 
    
    """
    
    #### Keep the first channel of the audio file only:
    if len(audio_array.shape)>1:
        audio = audio_array[:,0]
    else:
        audio = audio_array
        
    length = audio.size 
    
    #### Translate audio by 2^15 so as to make its dtype unsigned.
    audio = audio + 2**15
    
    audio_bin = np.zeros(shape=(length,17),dtype=int)
    for i in range(length):
        audio_bin[i,:] = int2bitarray(audio[i],17)
        
    return audio_bin
    
def Bin2Audio(audio_bin):
    
    """
    Converts a 17-bits binary audio array to an int16 1D-(one channel) audio_array.
    
    Parameters:
    - audio_bin: 17 bits binary array shaped (length,17). 
    
    returns:
    - int16 1D-audio-array of size = length.
    
    """
            
    length = audio_bin.shape[0] 
    
    audio = np.zeros(length,dtype=int)
    
    for i in range(length):
        audio[i] = bitarray2int(audio_bin[i,:])
    
    #### Translate audio by - 2^15 so as to make its dtype signed int16.

    audio = audio - 2**15

    return audio.astype(np.int16)
    
    