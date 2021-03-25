import math
import numpy as np
from kernels_for_sequence import *
from tqdm import tqdm
from utils import normalize_gram
from functools import partial
import itertools

# Linear kernel    
def Linear_kernel(x,y):
  return np.dot(x, y.T)

# see http://www.cs.toronto.edu/~duvenaud/cookbook/index.html 
def shifted_lineal_kernel(x, y, sig_b=1.2, sig_v=0.8):
    return sig_b**2 + (sig_v**2) * np.dot(x, y.T)
    
# RBF kernel
def RBF_kernel(x,y, gamma=10.): 
  # gamma = 1/nb_features by default
  return np.exp(-gamma * np.linalg.norm(x-y)**2)

# Another implementation (Vectorized version) of RBF kernel 
def RBF_kernel_vect(X, gamma=0.01): # gamma = 1/nb_features by default
  from scipy.spatial.distance  import pdist, squareform
  from scipy import exp
  sq_dists = pdist(X, 'sqeuclidean')  # pairwise distances     
  mat_sq_dists = squareform(sq_dists) # convert it to square matrix (symmetric)       
  K = exp(-gamma * mat_sq_dists)
  return K

# Quadratic kernel    
def Quadratic_kernel(x,y):
  return np.dot(x, y.T)**2

# rational quadratic kernel
# see https://arxiv.org/pdf/1302.4922.pdf
def r_quadratic_kernel(x, y, alpha=0.5, sig=0.8, l=1.):
    return (sig**2) * ((1+ np.dot(x-y, (x-y).T)/(2*alpha*l*l))**(-alpha))
    
# Polynomial kernel
def Polynomial_kernel(x,y, d=3, gamma=0.8):
  return (gamma + np.dot(x.T, y))**d

# Sigmoid Kernel
# http://www.raetschlab.org/lectures/ismb09tutorial/handout.pdf
def sigmoid_kernel(x, y, b=0.1, c=0.9):
    return np.tanh(c * x.dot(y.T) + b)

# Exponentiel cosine kernel -- Not working
def exp_cos_kernel(x, y, gamma=10., omega=10.):
    D = np.linalg.norm(x-y)**2
    return 0.5 * np.exp(-gamma * D**2) + 0.5 * np.cos(omega * D**2)
    
# see http://www.cs.toronto.edu/~duvenaud/cookbook/index.html   
# used for sequence 
def periodic_kernel(x, y, sigma=0.2, period=3.14, scale=10.):
    #from utils import seq_to_num
    #x = seq_to_num(s1)
    #y = seq_to_num(s2)
    return sigma**2 * np.exp(-2 * np.sin(np.pi * np.linalg.norm(x-y, ord=1) / period)** 2 / (scale**2))

# see http://www.cs.toronto.edu/~duvenaud/cookbook/index.html 
# used for sequence
def locally_periodic_kernel(x, y, sigma=1., period=1, scale=1.):
    return periodic_kernel(x, y, sigma, period, scale) * np.exp(-np.linalg.norm(x-y)/(2*scale*scale))

# Multiple Kernel Learning
# http://www.raetschlab.org/lectures/ismb09tutorial/handout.pdf
# Could be generalize to be a linear combination between n kernels
def MKL(x, y, K1, K2, t=0.2): # add two or more kernels
    return (1-t)*K1(x,y) + t*K2(x,y)
    
    
# Compute the Gram matrix  
def RBF_Gram_Matrix(X, Y, kernel="RBF", gamma=0.01, degree=2, shift=2, normalize=False,
                           gap=2, nplets=3):

    if kernel == "linear":
        ker = Linear_kernel
    elif kernel == "shift_linear":
        ker = shifted_lineal_kernel
    elif kernel == "quadratic":
        ker = Quadratic_kernel
    elif kernel == "r_quadratic":
        ker = r_quadratic_kernel
    elif kernel == "rbf":
        ker = partial(RBF_kernel, gamma=gamma)
    elif kernel == "sigmoid":
        ker = sigmoid_kernel
    elif kernel == "exp_cos":
        ker = exp_cos_kernel
    elif kernel == "periodic":
        ker = periodic_kernel
    elif kernel == "l_periodic":
        ker = locally_periodic_kernel
    elif kernel == "polynomial":
        ker = partial(Polynomial_kernel, d=degree, gamma=gamma)
    elif kernel == "MKL":
        ker = partial(MKL, K1=sigmoid_kernel, K2=RBF_kernel)
    #                       For strings
    elif kernel == "spectrum":
        ker = partial(spectrum_kernel, k=nplets)
    elif kernel == "substring":
        s = X[0][0]
        combinations = itertools.product(range(len(s) - nplets + 1), range(-shift, shift + 1))
        ker = partial(substring_kernel, k=nplets, delta=shift, combinations=combinations)
    elif kernel == "substring_mis":
        # s = X[0][0]
        # combinations = itertools.product(range(len(s) - nplets + 1), range(-shift, shift + 1))
        # ker = partial(substring_mismatch_kernel, combinations=combinations)
        ker = partial(substring_mismatch_kernel_fast, n=nplets, k=1, charset='ATCG', norm=normalize)

    elif kernel == "substring_mis_w":
        # s = X[0][0]
        # combinations = itertools.product(range(len(s) - nplets + 1), range(-shift, shift + 1))
        # ker = partial(w_substring_mismatch_kernel, k=nplets, delta=shift, combinations=combinations)
        ker = partial(substring_mismatch_kernel_wighted_fast, n=nplets, k=1, charset='ATCG')

    elif kernel == "substring_mis_mixed" or kernel == 'nlck_mismatch':
        ker1 = partial(substring_mismatch_kernel_wighted_fast, n=nplets+1, k=1, charset='ATCG')
        ker2 = partial(substring_mismatch_kernel_wighted_fast, n=nplets, k=1, charset='ATCG')
        ker3 = partial(substring_mismatch_kernel_wighted_fast, n=nplets-1, k=1, charset='ATCG')

    elif kernel == "wdk":
        ker = Weight_Degree_Kernel
    elif kernel == "wdkws":
        ker = partial(Weight_Degree_Kernel_w_Shifts, d=degree, S=shift)
    elif kernel == "gappy":
        ker = partial(gappy_kernel, k=nplets, g=gap)
    elif kernel == "MAK":
        ker = partial(mean_alignement_kernel, sigma=gamma)
    elif kernel == "LAK":
        ker = local_align_kernel
    elif kernel == "LAK_affine":
        ker = affine_align
    elif kernel == "needleman_wunsch":
        ker = Needleman_Wunsch
    elif kernel == "smith_waterman":
        ker = Smith_Waterman
    elif kernel == "string":
        ker = partial(string_kernel, lamda=gamma, k=nplets)
    else:
        raise NotImplemented



    if len(Y)==0:
        n = X.shape[0]
        gram_matrix = np.zeros((n, n), dtype=np.float32)

        if kernel == 'substring_mis':
            gram_matrix = ker(X, X)
            return gram_matrix
        elif kernel == 'substring_mis_w':
            gram_matrix = ker(X, X)
            return gram_matrix
        elif kernel == 'substring_mis_mixed':
            gram_matrix = 0.2*ker1(X, X) + 0.6*ker2(X, X) + 0.2*ker3(X, X)
            return gram_matrix
        elif kernel == 'nlck_mismatch':
            gram_matrices = [ker1(X, X), ker2(X, X), ker3(X, X)]
            return  gram_matrices

        for i in tqdm(range(n), desc="Computing Gram Matrix len-0"):
            #for j in tqdm(range(i,n), desc="Nested loop"):
            for j in range(i,n):
                if kernel == "wdk":
                    gram_matrix[i, j] = Weight_Degree_Kernel(X[i], X[j], degree, i, j)
                gram_matrix[i, j] = ker(X[i],X[j])
                gram_matrix[j,i] = gram_matrix[i,j] # + 0.0001
        if normalize: gram_matrix = normalize_gram(gram_matrix)
        return gram_matrix
    else:
        n = X.shape[0]
        len_X = X.shape[0]
        len_Y = Y.shape[0]
        gram_matrix = np.zeros((len_X, len_Y), dtype=np.float32)

        if kernel == 'substring_mis':
            gram_matrix = ker(X, Y)
            return gram_matrix.T
        elif kernel == 'substring_mis_w':
            gram_matrix = ker(X, Y)
            return gram_matrix.T
        elif kernel == 'substring_mis_mixed':
            gram_matrix = 0.2*ker1(X, Y) + 0.6*ker2(X, Y) + 0.2*ker3(X, Y)
            return gram_matrix.T
        elif kernel == 'nlck_mismatch':
            gram_matrices = [ker1(X, Y).T, ker2(X, Y).T, ker3(X, Y).T]
            return  gram_matrices

        for i in tqdm(range(len_X), desc="Computing Gram Matrix -"):
            #for j in tqdm(range(i,len_Y), desc="Nested loop"):
            for j in range(i,len_Y):
                for j in range(i, n):
                    if kernel == "wdk":
                        gram_matrix[i, j] = Weight_Degree_Kernel(X[i], Y[j], degree, i, j)
                    gram_matrix[i, j] = ker(X[i], Y[j])
                    gram_matrix[j, i] = gram_matrix[i, j]  # + 0.0001
        if normalize: gram_matrix = normalize_gram(gram_matrix)        
        return gram_matrix
