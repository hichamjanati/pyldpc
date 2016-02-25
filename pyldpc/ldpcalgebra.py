import math
import numpy as np
import scipy
pi = math.pi


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


def BinaryProduct(X,Y):
        
    """ Binary Matrices or Matrix-vector product in Z/2Z. Works with scipy.sparse.csr_matrix matrices X,Y too."""
 
    A = X.dot(Y)
    
    if type(A)!=scipy.sparse.csr_matrix:
        return A%2
    
    return A.toarray()%2

    

def GaussJordan(MATRIX,change=0):

    """ 
    Description:

    Performs the row reduced echelon form of MATRIX and returns it.

    If change = 1, all changes in the MATRIX's rows are applied to identity matrix P: 

    Let A be our parameter MATRIX. refA the reduced echelon form of A. P is the square invertible matrix:

    P.A = Aref.

    -------------------------------------------------
    Parameters: 

    MATRIX: 2D-Array. 
    change : boolean (default = 0)

    ------------------------------------------------

    change = 0  (default)
     >>> Returns 2D-Array Row Reduced Echelon form of Matrix

    change = 1 
    >>> Returns Tuple of 2D-arrays (refMATRIX, P) where P is described above.

    """

    A = np.copy(MATRIX)
    m,n = A.shape
    
    if change:
        P = np.identity(m).astype(int)

    pivot_old = -1 
    for j in range(n):            

        filtre_down = A[pivot_old+1:m,j]
        pivot = np.argmax(filtre_down)+pivot_old+1
        

        if A[pivot,j]:
            pivot_old+=1 
            if pivot_old != pivot:
                aux = np.copy(A[pivot,:])
                A[pivot,:] = A[pivot_old,:]
                A[pivot_old,:] = aux
                if change:
                    aux = np.copy(P[pivot,:])
                    P[pivot,:] = P[pivot_old,:]
                    P[pivot_old,:] = aux

            

            for i in range(m):
                if i!=pivot_old and A[i,j]:
                    if change:
                        P[i,:] = abs(P[i,:]-P[pivot_old,:])
                    A[i,:] = abs(A[i,:]-A[pivot_old,:])


        if pivot_old == m-1:
            break

 
    if change:    
        return A,P 
    return A 

    
    
def BinaryRank(MATRIX):
    """ Computes rank of a binary Matrix using Gauss-Jordan algorithm"""
    A = np.copy(MATRIX)
    m,n = A.shape
    
    A = GaussJordan(A)
    
    return sum([a.any() for a in A])

    
pi = math.pi
def f1(y,sigma):
    """ Normal Density N(1,sigma) """ 
    return(np.exp(-.5*pow((y-1)/sigma,2))/(sigma*math.sqrt(2*pi)))

def fM1(y,sigma):
    """ Normal Density N(-1,sigma) """ 

    return(np.exp(-.5*pow((y+1)/sigma,2))/(sigma*math.sqrt(2*pi)))

def Bits2i(H,i):
    """
    Computes list of elements of N(i)-j:
    List of variables (bits) connected to Parity node i.

    """
    if type(H)!=scipy.sparse.csr_matrix:
        m,n=H.shape
        return ([a for a in range(n) if H[i,a] ])
    
    indj = H.indptr
    indi = H.indices
    
    return [indi[a] for a in range(indj[i],indj[i+1])]


def Nodes2j(tH,j):
    
    """
    Computes list of elements of M(j):
    List of nodes (PC equations) connecting variable j.

    """
    
    return Bits2i(tH,j)

def BitsAndNodes(H):
    
    m,n = H.shape
    if type(H)==scipy.sparse.csr_matrix:
        tH = scipy.sparse.csr_matrix(np.transpose(H.toarray()))
    else:
        tH = np.transpose(H)
        
    Bits = [Bits2i(H,i) for i in range(m)]
    Nodes = [Nodes2j(tH,j)for j in range(n)]
    
    return Bits,Nodes
    
def InCode(H,x):

    """ Computes Binary Product of H and x. If product is null, x is in the code.

        Returns appartenance boolean. 
    """
        
    return  (BinaryProduct(H,x)==0).all()


def GaussElimination(MATRIX,B):

    """ Applies Gauss Elimination Algorithm to MATRIX in order to solve a linear system MATRIX.X = B. 
    MATRIX is transformed to row echelon form: 

         |1 * * * * * |
         |0 1 * * * * |
         |0 0 1 * * * |
         |0 0 0 1 * * | 
         |0 0 0 0 1 * |
         |0 0 0 0 0 1 |
         |0 0 0 0 0 0 |
         |0 0 0 0 0 0 |
         |0 0 0 0 0 0 |


    Same row operations are applied on 1-D Array vector B. Both arguments are sent back.
    
    --------------------------------------
    
    Parameters:
    
    MATRIX: 2D-array. 
    B:      1D-array. Size must equal number of rows of MATRIX.
            
    -----------------------------------
    Returns:
    
    Modified arguments MATRIX, B as described above.
    
         """

    A = np.copy(MATRIX)
    b = np.copy(B)
    n,k = A.shape


    if b.size != n:
        raise ValueError('Size of B must match number of rows of MATRIX to solve MATRIX.X = B')

    for j in range(min(k,n)):
        listeDePivots = [i for i in range(j,n) if A[i,j]]
        if len(listeDePivots)>0:
            pivot = np.min(listeDePivots)
        else:
            continue
        if pivot!=j:
            aux = np.copy(A[j,:])
            A[j,:] = A[pivot,:]
            A[pivot,:] = aux

            aux = np.copy(b[j])
            b[j] = b[pivot]
            b[pivot] = aux

        for i in range(j+1,n):
            if A[i,j]:
                A[i,:] = abs(A[i,:]-A[j,:])
                b[i] = abs(b[i]-b[j])

    return A,b

