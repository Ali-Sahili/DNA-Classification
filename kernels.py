import math
import numpy as np
from kernels_for_sequence import *
from tqdm import tqdm
from utils import normalize_gram

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
    
    if len(Y)==0:
        n = X.shape[0]
        gram_matrix = np.zeros((n, n), dtype=np.float32)

        for i in tqdm(range(n), desc="Computing Gram Matrix"):
            #for j in tqdm(range(i,n), desc="Nested loop"):
            for j in range(i,n):
                if kernel=="linear": gram_matrix[i,j] = Linear_kernel(X[i],X[j])
                elif kernel=="shift_linear": gram_matrix[i,j] = shifted_lineal_kernel(X[i],X[j])
                elif kernel=="quadratic": gram_matrix[i,j] = Quadratic_kernel(X[i],X[j])
                elif kernel=="r_quadratic": gram_matrix[i,j] = r_quadratic_kernel(X[i],X[j])
                elif kernel=="rbf": gram_matrix[i,j] = RBF_kernel(X[i],X[j],gamma)
                elif kernel=="sigmoid": gram_matrix[i,j] = sigmoid_kernel(X[i],X[j])
                elif kernel=="exp_cos": gram_matrix[i,j] = exp_cos_kernel(X[i],X[j])
                elif kernel=="periodic": gram_matrix[i,j] = periodic_kernel(X[i],X[j])
                elif kernel=="l_periodic": gram_matrix[i,j] = locally_periodic_kernel(X[i],X[j])
                elif kernel=="polynomial": gram_matrix[i,j] = Polynomial_kernel(X[i],X[j], 
                                                                               degree, gamma)
                elif kernel=="MKL": gram_matrix[i,j] = MKL(X[i],X[j], K1=sigmoid_kernel,
                                                                      K2=RBF_kernel)
                #                       For strings    
                elif kernel=="spectrum": gram_matrix[i,j] = spectrum_kernel(X[i],X[j], k=nplets)
                elif kernel=="substring":gram_matrix[i,j] = substring_kernel(X[i],X[j],k=nplets,
                                                                                 delta=shift)
                elif kernel=="substring_mis":gram_matrix[i,j] = substring_mismatch_kernel(X[i], 
                                                                 X[j], k=nplets, delta=shift)
                elif kernel=="substring_mis_w":gram_matrix[i,j] = w_substring_mismatch_kernel(
                                                             X[i], X[j], k=nplets, delta=shift)
                                                                                              
                elif kernel=="wdk":gram_matrix[i,j] = Weight_Degree_Kernel(X[i],X[j],degree,i,j)
                elif kernel=="wdkws":gram_matrix[i,j] = Weight_Degree_Kernel_w_Shifts(X[i],X[j],
                                                                         degree, shift)
                elif kernel=="gappy": gram_matrix[i,j] = gappy_kernel(X[i],X[j], nplets, gap)
                elif kernel=="MAK": gram_matrix[i,j] = mean_alignement_kernel(X[i], X[j], gamma)
                elif kernel=="LAK": gram_matrix[i,j] = local_align_kernel(X[i],X[j])
                elif kernel=="LAK_affine": gram_matrix[i,j] = affine_align(X[i],X[j])
                elif kernel=="needleman_wunsch": gram_matrix[i,j] = Needleman_Wunsch(X[i],X[j])
                elif kernel=="smith_waterman": gram_matrix[i,j] = Smith_Waterman(X[i],X[j])
                elif kernel=="string": gram_matrix[i,j] = string_kernel(gamma,nplets,X[i], X[j])
                else: raise NotImplemented
                gram_matrix[j,i] = gram_matrix[i,j] # + 0.0001
        if normalize: gram_matrix = normalize_gram(gram_matrix)
        return gram_matrix
    else:
        len_X = X.shape[0]
        len_Y = Y.shape[0]
        gram_matrix = np.zeros((len_X, len_Y), dtype=np.float32)
        
        for i in tqdm(range(len_X), desc="Computing Gram Matrix"):
            #for j in tqdm(range(i,len_Y), desc="Nested loop"):
            for j in range(i,len_Y):
                if kernel=="linear": gram_matrix[i,j] = Linear_kernel(X[i],Y[j])
                elif kernel=="shift_linear": gram_matrix[i,j] = shifted_lineal_kernel(X[i],Y[j])
                elif kernel=="quadratic": gram_matrix[i,j] = Quadratic_kernel(X[i],Y[j])
                elif kernel=="r_quadratic": gram_matrix[i,j] = r_quadratic_kernel(X[i],Y[j])
                elif kernel=="rbf": gram_matrix[i,j] = RBF_kernel(X[i],Y[j],gamma)
                elif kernel=="sigmoid": gram_matrix[i,j] = sigmoid_kernel(X[i],Y[j])
                elif kernel=="exp_cos": gram_matrix[i,j] = exp_cos_kernel(X[i],Y[j])
                elif kernel=="periodic": gram_matrix[i,j] = periodic_kernel(X[i],Y[j])
                elif kernel=="l_periodic": gram_matrix[i,j] = locally_periodic_kernel(X[i],Y[j])
                elif kernel=="polynomial": gram_matrix[i,j] = Polynomial_kernel(X[i],Y[j], 
                                                                               degree, gamma)
                elif kernel=="MKL": gram_matrix[i,j] = MKL(X[i],Y[j], K1=sigmoid_kernel,
                                                                      K2=RBF_kernel)
                
                #                       For strings    
                elif kernel=="spectrum": gram_matrix[i,j] = spectrum_kernel(X[i],Y[j], k=nplets)
                elif kernel=="substring":gram_matrix[i,j] = substring_kernel(X[i],Y[j],k=nplets,
                                                                                 delta=shift)
                elif kernel=="substring_mis":gram_matrix[i,j] = substring_mismatch_kernel(X[i], 
                                                                 Y[j], k=nplets, delta=shift)
                elif kernel=="substring_mis_w":gram_matrix[i,j] = w_substring_mismatch_kernel(
                                                             X[i], Y[j], k=nplets, delta=shift)
                elif kernel=="wdk":gram_matrix[i,j] = Weight_Degree_Kernel(X[i],Y[j],degree,i,j)
                elif kernel=="wdkws":gram_matrix[i,j] = Weight_Degree_Kernel_w_Shifts(X[i],Y[j],
                                                                         degree, shift)
                elif kernel=="gappy": gram_matrix[i,j] = gappy_kernel(X[i],Y[j], nplets, gap)
                elif kernel=="MAK": gram_matrix[i,j] = mean_alignement_kernel(X[i], Y[j], gamma)
                elif kernel=="LAK": gram_matrix[i,j] = local_align_kernel(X[i],Y[j])
                elif kernel=="LAK_affine": gram_matrix[i,j] = affine_align(X[i],Y[j])
                elif kernel=="needleman_wunsch": gram_matrix[i,j] = Needleman_Wunsch(X[i],Y[j])
                elif kernel=="smith_waterman": gram_matrix[i,j] = Smith_Waterman(X[i],Y[j])
                elif kernel=="string": gram_matrix[i,j] = string_kernel(gamma,nplets,X[i], Y[j])
                else: raise NotImplemented
        if normalize: gram_matrix = normalize_gram(gram_matrix)        
        return gram_matrix
