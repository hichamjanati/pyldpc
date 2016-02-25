import numpy as np
from .ldpcalgebra import*
import scipy

__all__ = ['BinaryProduct', 'InCode', 'BinaryRank','Decoding_logBP','Decoding_BP','DecodedMessage']
def Decoding_BP(H,y,SNR,max_iter=1):

    """ Decoding function using Belief Propagation algorithm.
        
        
        IMPORTANT: H can be scipy.sparse.csr_matrix object to speed up calculations if n > 1000 highly recommanded. 
    -----------------------------------
    Parameters:
    
    H: 2D-array (OR scipy.sparse.csr_matrix object) Parity check matrix, shape = (m,n) 

    y: n-vector recieved after transmission in the channel. (In general, returned 
    by Coding Function)


    Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN used in coding.
    
    max_iter: (default = 1) max iterations of the main loop. Increase if decoding is not error-free.

     """
        
    m,n=H.shape
    if not len(y)==n:
        raise ValueError('Size of y must be equal to number of parity matrix\'s columns n')

    if m>=n:
        raise ValueError('H must be of shape (m,n) with m < n')

    sigma = 10**(-SNR/20)
    
    p0 = np.zeros(shape=n)
    p0 = f1(y,sigma)/(f1(y,sigma) + fM1(y,sigma))
    p1 = np.zeros(shape=n)
    p1 = fM1(y,sigma)/(f1(y,sigma) + fM1(y,sigma))


    #### ETAPE 0 : Initialization 
    q0 = np.zeros(shape=(m,n))
    q0[:] = p0

    q1 = np.zeros(shape=(m,n))
    q1[:] = p1

    r0 = np.zeros(shape=(m,n))
    r1 = np.zeros(shape=(m,n))

    count=0
    prod = np.prod
    
    Bits,Nodes = BitsAndNodes(H)

    
    while(True):

        count+=1 #Compteur qui empêche la boucle d'être infinie .. 

        #### ETAPE 1 : Horizontale

        deltaq = q0 - q1
        deltar = r0 - r1 
        

        for i in range(m):
            Ni=Bits[i]
            for j in Ni:
                Nij = Ni.copy()

                if j in Nij: Nij.remove(j)
                deltar[i,j] = prod(deltaq[i,Nij])
                

        r0 = 0.5*(1+deltar)
        r1 = 0.5*(1-deltar)


        #### ETAPE 2 : Verticale

        for j in range(n):
            Mj = Nodes[j]
            for i in Mj:
                Mji = Mj.copy()
                if i in Mji: Mji.remove(i)
                    
                q0[i,j] = p0[j]*prod(r0[Mji,j])
                q1[i,j] = p1[j]*prod(r1[Mji,j])
                
                if q0[i,j] + q1[i,j]==0:
                    q0[i,j]=0.5
                    q1[i,j]=0.5
              
                else:
                    alpha=1/(q0[i,j]+q1[i,j]) #Constante de normalisation alpha[i,j] 

                    q0[i,j]*= alpha
                    q1[i,j]*= alpha # Maintenant q0[i,j] + q1[i,j] = 1


        #### Calcul des probabilites à posteriori:
        q0_post = np.zeros(n)
        q1_post = np.zeros(n)
        
        for j in range(n):
            Mj=Nodes[j]
            q0_post[j] = p0[j]*prod(r0[Mj,j])
            q1_post[j] = p1[j]*prod(r1[Mj,j])
            
            if q0_post[j] + q1_post[j]==0:
                q0_post[j]=0.5
                q1_post[j]=0.5
                
            alpha = 1/(q0_post[j]+q1_post[j])
            
            q0_post[j]*= alpha
            q1_post[j]*= alpha
        

        x = np.array(q1_post > q0_post).astype(int)
        
        if InCode(H,x) or count >= max_iter:  
            break
  
    return x




def Decoding_logBP(H,y,SNR,max_iter=1):
    
    """ Decoding function using Belief Propagation algorithm (logarithmic version)

    IMPORTANT: if H is large (n>1000), H should be scipy.sparse.csr_matrix object to speed up calculations
    (highly recommanded. )
    -----------------------------------
    Parameters:
    
    H: 2D-array (OR scipy.sparse.csr_matrix object) Parity check matrix, shape = (m,n) 

    y: n-vector recieved after transmission in the channel. (In general, returned 
    by Coding Function)


    Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN used in coding.
    
    max_iter: (default = 1) max iterations of the main loop. Increase if decoding is not error-free.

     """
        
    m,n=H.shape

    if not len(y)==n:
        raise ValueError('La taille de y doit correspondre au nombre de colonnes de H')

    if m>=n:
        raise ValueError('H doit avoir plus de colonnes que de lignes')
    
    
    var = 10**(-SNR/10)

    ### ETAPE 0: initialisation 
    
    Lc = 2*y/var

    Lq=np.zeros(shape=(m,n))

    Lr = np.zeros(shape=(m,n))
    
    count=0
    
    prod=np.prod
    tanh = np.tanh
    log = np.log
    
    	
    Bits,Nodes = BitsAndNodes(H)
    while(True):

        count+=1 #Compteur qui empêche la boucle d'être infinie .. 

        #### ETAPE 1 : Horizontale
        for i in range(m):
            Ni = Bits[i]
            for j in Ni:
                Nij = Ni.copy()

                if j in Nij: Nij.remove(j)
            
                if count==1:
                    X = prod(tanh(0.5*Lc[Nij]))
                else:
                    X = prod(tanh(0.5*Lq[i,Nij]))
                num = 1 + X
                denom = 1 - X
                if num == 0: 
                    Lr[i,j] = -1
                elif denom  == 0:
                    Lr[i,j] =  1
                else: 
                    Lr[i,j] = log(num/denom)
              

        #### ETAPE 2 : Verticale
        for j in range(n):
            Mj = Nodes[j]
            
            for i in Mj:
                Mji = Mj.copy()
                if i in Mji: Mji.remove(i)

                Lq[i,j] = Lc[j]+sum(Lr[Mji,j])
        
 
        #### LLR a posteriori:
        L_posteriori = np.zeros(n)
        for j in range(n):
            Mj = Nodes[j]

            L_posteriori[j] = Lc[j] + sum(Lr[Mj,j])

        x = np.array(L_posteriori <= 0).astype(int)
            
        product = InCode(H,x)

        if product or count >= max_iter:  
            break
        
    return x
    
    
def Decoding_BP_ext(H,BitsNodesTuple,y,SNR,max_iter=1):

    """ Decoding function using Belief Propagation algorithm.
        
        
        IMPORTANT: H can be scipy.sparse.csr_matrix object to speed up calculations if n > 1000 highly recommanded. 
    -----------------------------------
    Parameters:
    
    H: 2D-array (OR scipy.sparse.csr_matrix object) Parity check matrix, shape = (m,n) 

    BitsNodesTuple: Tuple returned by BitsAndNodes function. 

    y: n-vector recieved after transmission in the channel. (In general, returned 
    by Coding Function)


    Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN used in coding.
    
    max_iter: (default = 1) max iterations of the main loop. Increase if decoding is not error-free.

     """
        
    m,n=H.shape
    if not len(y)==n:
        raise ValueError('Size of y must be equal to number of parity matrix\'s columns n')

    if m>=n:
        raise ValueError('H must be of shape (m,n) with m < n')

    sigma = 10**(-SNR/20)
    
    p0 = np.zeros(shape=n)
    p0 = f1(y,sigma)/(f1(y,sigma) + fM1(y,sigma))
    p1 = np.zeros(shape=n)
    p1 = fM1(y,sigma)/(f1(y,sigma) + fM1(y,sigma))



    #### ETAPE 0 : Initialization 
    q0 = np.zeros(shape=(m,n))
    q0[:] = p0

    q1 = np.zeros(shape=(m,n))
    q1[:] = p1

    r0 = np.zeros(shape=(m,n))
    r1 = np.zeros(shape=(m,n))

    count=0
    prod = np.prod
    
    Bits = BitsNodesTuple[0]
    Nodes = BitsNodesTuple[1]
    
    while(True):

        count+=1 #Compteur qui empêche la boucle d'être infinie .. 

        #### ETAPE 1 : Horizontale

        deltaq = q0 - q1
        deltar = r0 - r1 
        

        for i in range(m):
            Ni=Bits[i]
            for j in Ni:
                Nij = Ni.copy()

                if j in Nij: Nij.remove(j)
                deltar[i,j] = prod(deltaq[i,Nij])
                

        r0 = 0.5*(1+deltar)
        r1 = 0.5*(1-deltar)


        #### ETAPE 2 : Verticale

        for j in range(n):
            Mj = Nodes[j]
            for i in Mj:
                Mji = Mj.copy()
                if i in Mji: Mji.remove(i)
                    
                q0[i,j] = p0[j]*prod(r0[Mji,j])
                q1[i,j] = p1[j]*prod(r1[Mji,j])
                
                if q0[i,j] + q1[i,j]==0:
                    q0[i,j]=0.5
                    q1[i,j]=0.5
              
                else:
                    alpha=1/(q0[i,j]+q1[i,j]) #Constante de normalisation alpha[i,j] 

                    q0[i,j]*= alpha
                    q1[i,j]*= alpha # Maintenant q0[i,j] + q1[i,j] = 1


        #### Calcul des probabilites à posteriori:
        q0_post = np.zeros(n)
        q1_post = np.zeros(n)
        
        for j in range(n):
            Mj=Nodes[j]
            q0_post[j] = p0[j]*prod(r0[Mj,j])
            q1_post[j] = p1[j]*prod(r1[Mj,j])
            
            if q0_post[j] + q1_post[j]==0:
                q0_post[j]=0.5
                q1_post[j]=0.5
                
            alpha = 1/(q0_post[j]+q1_post[j])
            
            q0_post[j]*= alpha
            q1_post[j]*= alpha
        

        x = np.array(q1_post > q0_post).astype(int)
        
        if InCode(H,x) or count >= max_iter:  
            break
  
    return x




def Decoding_logBP_ext(H,BitsNodesTuple,y,SNR,max_iter=1):
    
    """ Decoding function using Belief Propagation algorithm (logarithmic version)

    IMPORTANT: if H is large (n>1000), H should be scipy.sparse.csr_matrix object to speed up calculations
    (highly recommanded. )
    -----------------------------------
    Parameters:
    
    H: 2D-array (OR scipy.sparse.csr_matrix object) Parity check matrix, shape = (m,n) 

    BitsNodesTuple: Tuple returned by BitsAndNodes function. 

    y: n-vector recieved after transmission in the channel. (In general, returned 
    by Coding Function)


    Signal-Noise Ratio: SNR = 10log(1/variance) in decibels of the AWGN used in coding.
    
    max_iter: (default = 1) max iterations of the main loop. Increase if decoding is not error-free.

     """
        
    m,n=H.shape

    if not len(y)==n:
        raise ValueError('La taille de y doit correspondre au nombre de colonnes de H')

    if m>=n:
        raise ValueError('H doit avoir plus de colonnes que de lignes')
    
    
    var = 10**(-SNR/10)

    ### ETAPE 0: initialisation 
    
    Lc = 2*y/var

    Lq=np.zeros(shape=(m,n))

    Lr = np.zeros(shape=(m,n))
    
    count=0
    
    prod=np.prod
    tanh = np.tanh
    log = np.log
    
    Bits = BitsNodesTuple[0]
    Nodes = BitsNodesTuple[1]
    while(True):

        count+=1 #Compteur qui empêche la boucle d'être infinie .. 

        #### ETAPE 1 : Horizontale
        for i in range(m):
            Ni = Bits[i]
            for j in Ni:
                Nij = Ni.copy()

                if j in Nij: Nij.remove(j)
            
                if count==1:
                    X = prod(tanh(0.5*Lc[Nij]))
                else:
                    X = prod(tanh(0.5*Lq[i,Nij]))
                num = 1 + X
                denom = 1 - X
                if num == 0: 
                    Lr[i,j] = -1
                elif denom  == 0:
                    Lr[i,j] =  1
                else: 
                    Lr[i,j] = log(num/denom)
              

        #### ETAPE 2 : Verticale
        for j in range(n):
            Mj = Nodes[j]
            
            for i in Mj:
                Mji = Mj.copy()
                if i in Mji: Mji.remove(i)

                Lq[i,j] = Lc[j]+sum(Lr[Mji,j])
        
 
        #### LLR a posteriori:
        L_posteriori = np.zeros(n)
        for j in range(n):
            Mj = Nodes[j]

            L_posteriori[j] = Lc[j] + sum(Lr[Mj,j])

        x = np.array(L_posteriori <= 0).astype(int)
            
        product = InCode(H,x)

        if product or count >= max_iter:  
            break
        
    return x

def DecodedMessage(tG,x):

    """
    Let G be a coding matrix. tG its transposed matrix. x a n-vector received after decoding.
    DecodedMessage Solves the equation on k-bits message v:  x = v.G => G'v'= x' by applying GaussElimination on G'.
    
    -------------------------------------
    
    Parameters:
    
    tG: Transposed Coding Matrix. Must have more rows than columns to solve the linear system. Must be full rank.
    x: n-array. Must be in the Code (in Ker(H)). 

    """
    n,k = tG.shape 
    
    if n < k:
        raise ValueError('Coding matrix G must have more columns than rows to solve the linear system on v\': G\'v\' = x\'')
    
                         
    rtG, rx = GaussElimination(tG,x)
    
    rank = sum([a.any() for a in rtG])
    if rank!= k:
        raise ValueError('Coding matrix G must have full rank = k to solve G\'v\' = x\'')
            
    message=np.zeros(k).astype(int)

    message[k-1]=rx[k-1]
    for i in reversed(range(k-1)):
        message[i]=abs(rx[i]-BinaryProduct(rtG[i,list(range(i+1,k))],message[list(range(i+1,k))]))

    return message

