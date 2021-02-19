from scipy.linalg import eigh
import numpy as np 
from kernels import RBF_Gram_Matrix

def rbf_kernel_pca(X, kernel="RBF", gamma=0.01, degree=3, n_components=10):       
    
    K = RBF_Gram_Matrix(X, [], kernel, gamma, degree)
            
    # Center the Gram matrix.    
    N = K.shape[0]
    one_n = np.ones((N,N)) / N    
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n) 
           
    # Obtaining eigenpairs from the centered Gram matrix    
    # scipy.linalg.eigh returns them in ascending order    
    eigvals, eigvecs = eigh(K)    
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
            
    # Collect the top k eigenvectors (projected examples)    
    X_pc = np.column_stack([eigvecs[:, i] for i in range(n_components)])        
    return X_pc
