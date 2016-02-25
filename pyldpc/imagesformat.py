import numpy as np

__all__=['int2bitarray','bitarray2int','Bin2Gray','Gray2Bin','RGB2Bin','Bin2RGB']

def int2bitarray(N,k):
    """
    Changes array's base from int (base 10) to binary (base 2)
    
    Parameters:
    
    N: int N 
    k: Width of the binary array you would like to change N into. N must not be greater than 2^k - 1. 
    
    >> Examples: int2bitarray(6,3) returns [1, 1, 0]
                 int2bitarray(6,5) returns [0, 0, 1, 1,0]
                 int2bitarray(255,8) returns [1, 1, 1, 1, 1, 1, 1, 1]
                 int2bitarray(255,10) returns [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]


    """

    binary_string = bin(N) 
    length = len(binary_string)
    bitarray = np.zeros(k, 'int')
    for i in range(length-2):
        bitarray[k-i-1] = int(binary_string[length-i-1])

    return bitarray

def bitarray2int(bitarray):
    
    """ Changes array's base from binary (base 2) to int (base 10).
    
    Parameters:
    
    bitarray: Binary Array.

    >> Examples: bitarray2int([1, 1, 0]) returns 6
                 bitarray2int([0, 0, 1, 1,0]) returns 6
                 bitarray2int([1, 1, 1, 1, 1, 1, 1, 1]) returns 255
                 

    
    """
    
    bitstring = "".join([str(i) for i in bitarray])
    
    return int(bitstring,2)
    
def Gray2Bin(img):
    """ Puts a GrayScale Image on a binary form 
    
    Parameters:
    
    img_array: 2-D array of a grayscale image (no 3rd dimension)
    
    returns:
    
    3-D img_array in a binary form, each pixel uint8 is transformed to an 8-bits array
    
    
    >>> Example:  the grayscale (2x2) image [[2, 127],    
                                             [255, 0]]
                                        
    will be conveterted to the (2x2x8) binary image:   [[[0, 0, 0, 0, 0, 0, 1, 0],[0, 1, 1, 1, 1, 1, 1, 1]],
                                                        [[1, 1, 1, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 0, 0, 0]]]
                                                       
                                                       
    """
    if not len(img.shape)==2:
        raise ValueError('{} must have 2 dimensions. Make sure it\'s a grayscale image.')
        
    height,width = img.shape
    
    img_bin = np.zeros(shape=(height,width,8),dtype=int)
    
    for i in range(height):
        for j in range(width):
            img_bin[i,j,:] = int2bitarray(img[i,j],8)
            
    return img_bin
    
    
def Bin2Gray(img_bin):
    
    """ Puts a 8-bits binary Image to uint8
    
    Parameters:
    
    img_array: 3-D array (height, width, 8)
    
    returns:
    
    2-D img_array in grayscale
    
    >>> Example:  the (2x2x8) binary image:   [[[0, 0, 0, 0, 0, 0, 1, 0],[0, 1, 1, 1, 1, 1, 1, 1]],
                                                          [[1, 1, 1, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 0, 0, 0]]] 
                                            
                                        
    will be conveterted to the (2x2) uint8 image [[2, 127],
                                                  [255, 0]]
    
    """
    height,width,k = img_bin.shape
    img_grayscale = np.zeros(shape=(height,width),dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            img_grayscale[i,j] = bitarray2int(img_bin[i,j,:])
            
    return img_grayscale
    
    
def RGB2Bin(img):
    """ Puts an RGB Image on a binary form 
    
    Parameters:
    
    img_array: 3-D array of an RGB image ( 3rd dimension = 3)
    
    returns:
    
    3-D img_array in a binary form, each pixel is transformed to an 24-bits binary array.
    
    
    >>> Example:  the grayscale (2x1x3) image [[[2, 127,0]],
                                               [[255, 0,1]]]
                                        
    will be conveterted to the (2x1x24) binary image:   
    [[[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]
                                                       
                                                       
    """
    
    height,width,depth = img.shape

    if not depth==3:
        raise ValueError('{}\'s 3rd dimension must be equal to 3 (RGB). Make sure it\'s an RGB image.')
      
    
    img_bin = np.zeros(shape=(height,width,24),dtype=int)
    
    for i in range(height):
        for j in range(width):
            R = int2bitarray(img[i,j,0],8)
            G = int2bitarray(img[i,j,1],8)
            B = int2bitarray(img[i,j,2],8)

            img_bin[i,j,:] = np.concatenate((R,G,B))
            
    return img_bin
    

def Bin2RGB(img_bin):
    
    """ Puts a 24-bits binary Image to 3xuint8 (RGB)
    
    Parameters:
    
    img_array: 3-D array (height, width, 24)
    
    returns:
    
    3-D img_array in RGB (height, width, 3)
    
    >>> Example:  the (2x1x24) binary image:   
    
   [[[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
   [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]
  
                                        
    will be conveterted to the (2x1x3) RGB image :
    
    [[[2, 127,0]],
    [[255, 0,1]]]

    """
    
    height,width,depth = img_bin.shape
    img_rgb = np.zeros(shape=(height,width,3),dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            R = bitarray2int(img_bin[i,j,:8])
            G = bitarray2int(img_bin[i,j,8:16])
            B = bitarray2int(img_bin[i,j,16:])

            img_rgb[i,j] = np.array([R,G,B],dtype=np.uint8)
            
    return img_rgb
    
