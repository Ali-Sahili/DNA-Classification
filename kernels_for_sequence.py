import numpy as np
import itertools
import math
from utils import seq_to_num

# slides resume almost all the below kernels
# http://www.raetschlab.org/lectures/ismb09tutorial/handout.pdf
# I add specific links for some kernels with additional explanations or python-codes

# Fisher scores: 
# https://proceedings.neurips.cc/paper/2002/file/12b1e42dc0746f22cf361267de07073f-Paper.pdf

#-----------------------------------------------------------------------------------
#                               Spectrum Kernel
#        https://www.ics.uci.edu/~welling/teatimetalks/kernelclub04/spectrum.pdf
#-----------------------------------------------------------------------------------
# Spectrum Kernel
def make_phi(seq, k=3):
    assert (len(seq) >= k)
    dict_ = {}
    for i in range(len(seq) - k + 1):
        target = seq[i : i + k]
        if target not in dict_.keys():
            dict_[target] = 1
        else:
            dict_[target] += 1
    return dict_

def spectrum_kernel(seq1, seq2, k=3):
    dict1 = make_phi(seq1, k)
    dict2 = make_phi(seq2, k)
    
    keys = list(set(dict1.keys()) & set(dict2.keys()))
    output = 0
    for key in keys:
        output += dict1[key] * dict2[key]
    return output
		
		
#-----------------------------------------------------------------------------------
#                               Substring Kernel
# https://mstrazar.github.io/tutorial/python/machine-learning/2018/08/31/string-kernels.html
#-----------------------------------------------------------------------------------
# Substring Kernel
def substring_kernel(s, t, k=3, delta=0):
    """ substring length: 𝑘 and shift: delta """
    assert(len(s)==len(t))
    L = len(s)
    return sum(((s[i:i + k] == t[d + i:d + i + k])
                for i, d in itertools.product(range(L - k + 1), range(-delta, delta + 1))
                if i + d + k <= L and i + d >= 0))



#-----------------------------------------------------------------------------------
#                      Substring Kernel with Mismatches
# https://mstrazar.github.io/tutorial/python/machine-learning/2018/08/31/string-kernels.html
#-----------------------------------------------------------------------------------
# Substring Kernel with Mismatches
def miss(s, t):
    """ Count the number of mismatches between two strings."""
    return sum((si != tj for si, tj in zip(s, t)))

def substring_mismatch_kernel(s, t, k=3, delta=1, m=1):
    """ String kernel with displacement and mismatches. """
    assert(len(s)==len(t))
    L = len(s)
    return sum(((miss(s[i:i + k], t[d + i:d + i + k]) <= m)
                for i, d in itertools.product(range(L - k + 1), range(-delta, delta + 1))
                if i + d + k <= L and i + d >= 0))


#-----------------------------------------------------------------------------------
#                  Weighted Substring Kernel with Mismatches
# https://mstrazar.github.io/tutorial/python/machine-learning/2018/08/31/string-kernels.html
#-----------------------------------------------------------------------------------
# Weighted Substring Kernel with Mismatches
def w_substring_mismatch_kernel(s, t, k=3, delta=1, m=1, gamma=0.99):
    """ String kernel with displacement, mismatches and exponential decay. """
    assert(len(s)==len(t))
    L = len(s)
    return sum(((np.exp(-gamma * d**2) \
                 * np.exp(-gamma * miss(s[i:i + k], t[d + i:d + i + k])) \
                 * (miss(s[i:i + k], t[d + i:d + i + k]) <= m) 
                for i, d in itertools.product(range(L - k + 1), range(-delta, delta + 1))
                if i + d + k <= L and i + d >= 0)))

             
#-----------------------------------------------------------------------------------
#                               Weight Degree Kernel
#                        https://github.com/mstrazar/mklaren
#-----------------------------------------------------------------------------------
# Weight Degree Kernel
def Weight_Degree_Kernel(x, y, d, i, j):

    def weights(d, k):  # d: int, maximal degree  # k: int, current degree
        return 2 * (d - k + 1) / d / (d + 1)
    
    L = len(x)
    
    if i==j:
        return (L - 1 + (1 - d) / 3)
    else:
        c_t = 0
        for k in range(1, d + 1):
            weights_k = weights(d, k)
            c_st = 0
            for l in range(1, L - k + 1):
                c_st += (x[l:l + k] == y[l:l + k])
            c_t += weights_k * c_st
        return c_t


#-----------------------------------------------------------------------------------
#                          Weight Degree Kernel with Shifts
#                         https://github.com/mstrazar/mklaren
#-----------------------------------------------------------------------------------
# Weight Degree Kernel with Shifts
def Weight_Degree_Kernel_w_Shifts(x, y, d, S): # d: maximal degree, S: maximal shift
    def weights(d, k):
        return 2 * (d - k + 1) / d / (d + 1)

    def delta(s):
        return 1/2/(s+1)
    
    L = len(x)
    
    c_t = 0
    for k in range(1, d + 1):
        weights_k = weights(d, k)
        c_st = 0
        for i in range(1, L - k + 1):
            for s in range(0, S+1):
                if s+i < L:
                    c_st += delta(s) * ((x[i+s:i+s+k] == y[i:i+k]) + (x[i:i+k] == y[i+s:i+s+k]))
        c_t += weights_k * c_st
    return c_t


#-----------------------------------------------------------------------------------
#                                     Gappy Kernel 
#-----------------------------------------------------------------------------------
# Gappy Kernel --> very slow
def gappy_k(x, k, g):
    """
    Compute feature vector of sequence x for Gappy Kernel (k, g)
    :param g: int, gap
    """
    betas = np.array([seq_to_num(''.join(c)) for c in itertools.product('ACGT', repeat=k)])
    phi = np.zeros(len(betas))
    gap_set = sum([list(itertools.combinations(x[i:i+k], k-g)) for i in range(100 - k)], [])
    for i, b in enumerate(betas):
        phi[i] = (b.all() in gap_set)
    return phi

def gappy_kernel(x,y,k,g):
    x = seq_to_num(x)
    y = seq_to_num(y)
    return np.dot(gappy_k(x, k, g), gappy_k(y, k, g))


#-----------------------------------------------------------------------------------
#                           One sided mean kernel (Mean Alignement)
#            https://www.sciencedirect.com/science/article/pii/S0925231215003665
#-----------------------------------------------------------------------------------
def reducedDistanceMatrix(seq1, seq2):

    n1, n2 = len(seq1), len(seq2)
    
    if n1>n2: seqA, seqB = seq2, seq1
    else: seqA, seqB = seq1, seq2
    
    nA, nB = len(seqA), len(seqB)
    
    reducedDistMat = np.zeros((nA, nB - nA + 1))
    for i in range(nA):
        for j in range(nB - nA + 1):
            reducedDistMat[i,j] = (seqA[i] - seqB[j + i])**2 #replace with any Hilbertian dist
    
    return reducedDistMat

def one_sided_mean(reducedDistMat):

    L, M = np.shape(reducedDistMat)
    dtwM = np.zeros((L + 1, M + 1))
    
    for m in range(M + 1):
        dtwM[0, m] = 0
    for l in range(L + 1):
        dtwM[l, 0] = 0
        
    dtwM[1,1] = reducedDistMat[0,0]

    for l in range(1,L+1):
        for m in range(1, M+1):
            if not (l, m)==(1,1):
                A = (float(l) - 1)/(l + m - 2) * dtwM[l-1, m]
                B = (float(m) - 1)/(l + m - 2) * dtwM[l, m-1]
                dtwM[l, m] = reducedDistMat[l - 1, m - 1] + A + B
    
    return dtwM[-1, -1]

def mean_alignement_kernel(seq1, seq2, sigma=3.):
    l1 = len(seq1)
    l2 = len(seq2)
    
    s1 = seq_to_num(seq1)
    s2 = seq_to_num(seq2)
    
    reducedMat = reducedDistanceMatrix(s1, s2)
    mean_dist = one_sided_mean(reducedMat)/max(l1,l2)
    return math.exp(-mean_dist/(2*sigma)) 
    
#-----------------------------------------------------------------------------------
#                          Local Alignment Kernel:  -- very slow
#   useful links: - https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/local.pdf
#                 - http://readiab.org/book/0.1.3/2/1 
#-----------------------------------------------------------------------------------

def local_align_kernel(x, y, gap=-7, match=10, mismatch=-5):
    # create a zero-filled matrix    
    A = [[0]*(len(y) + 1) for i in range(len(x) + 1)]  
    
    # fill in A in the right order
    for i in range(1, len(y)):
        for j in range(1, len(x)):           
            # Compute score: 
            # The score is based on the up, left, and upper-left neighbors            
            A[i][j] = max( A[i][j-1] + gap, A[i-1][j] + gap, 
                           A[i-1][j-1]+(match if x[i] == y[j] else mismatch), 0)
                
    return np.max(np.array(A))

# http://members.cbio.mines-paristech.fr/~jvert/publi/04kmcbbook/saigo.pdf
# page 8,9
# parameters e, d and beta are chosen according to the experiments done (link above)
# substitution matrix extracted from BLOSUM62
S = np.array([[4, 0, 0, 0], [0, 9, -3, -1], [0, -3, 6, 2], [0, -1, -2, 5]])


def affine_align(x, y, e=11, d=1, beta=0.5):
    # g(n) is linear g(0)=0 and g(n)=e+d(n-1)

    x, y = seq_to_num(x)-1, seq_to_num(y)-1
    n_x, n_y = len(x), len(y)
    M, X, Y, X2, Y2 = [np.zeros((n_x + 1, n_y + 1))]*5
    for i in range(1, n_x):
        for j in range(1, n_y):
            M[i, j] = np.exp(beta * S[x[i], y[j]]) * (1 + X[i-1, j-1]+Y[i-1, j-1] + M[i-1, j-1])
            X[i, j] = np.exp(beta * d) * M[i-1, j] + np.exp(beta * e) * X[i-1, j]
            Y[i, j] = np.exp(beta * d) * (M[i, j-1] + X[i, j-1]) + np.exp(beta * e) * Y[i, j-1]
            X2[i, j] = M[i-1, j] + X2[i-1, j]
            Y2[i, j] = M[i, j-1] + X2[i, j-1] + Y2[i, j-1]
    return (1/beta) * np.log(1 + X2[n_x, n_y] + Y2[n_x, n_y] + M[n_x, n_y])


# For Needleman-Wunsch this will be the last cell of the matrix,
# While for Smith-Waterman will be the cell with the highest score.
def Needleman_Wunsch(seq1, seq2, e=11, d=1, beta=0.5):
    m, n = len(seq1), len(seq2)
    M = np.zeros((m + 1, n + 1))
    X = np.zeros((m + 1, n + 1))
    Y = np.zeros((m + 1, n + 1))
    X2 = np.zeros((m + 1, n + 1))
    Y2 = np.zeros((m + 1, n + 1))

    assert(S.shape[0]==4)

    my_dict = {"A" : 0, "T" : 1, "C" : 2, "G" : 3}

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            M[i, j] = np.exp(beta * S[my_dict[seq1[i - 1]], my_dict[seq2[j - 1]]]) * (1 + X[i-1, j-1] + Y[i-1, j-1] + M[i-1, j-1])
            X[i, j] = np.exp(beta * d) * M[i - 1, j] + np.exp(beta * e) * X[i - 1, j]
            Y[i, j] = np.exp(beta * d) * (M[i, j-1] + X[i, j-1]) + np.exp(beta * e) * Y[i, j-1]
            X2[i, j] = M[i - 1, j] + X2[i - 1, j]
            Y2[i, j] = M[i, j - 1] + X2[i, j - 1] + Y2[i, j - 1]
    return 1 + X2[m, n] + Y2[m, n] + M[m, n]

def Smith_Waterman(x, y, e=11, d=1, beta=0.5):

    x, y = seq_to_num(x) - 1, seq_to_num(y) - 1
    n_x, n_y = len(x), len(y)
    M, X, Y, X2, Y2 = [np.zeros((n_x + 1, n_y + 1))] * 5
    for i in range(1, n_x):
        for j in range(1, n_y):
            M[i, j] = np.exp(beta * S[x[i],y[j]]) * max(1, X[i-1, j-1], Y[i-1, j-1], M[i-1, j-1])
            X[i, j] = max(np.exp(beta * d) * M[i - 1, j], np.exp(beta * e) * X[i - 1, j])
            Y[i, j] =max(np.exp(beta*d)*M[i,j-1],np.exp(beta*d)*X[i,j-1],np.exp(beta*e)*Y[i,j-1])
            X2[i, j] = max(M[i - 1, j], X2[i - 1, j])
            Y2[i, j] = max(M[i, j - 1], X2[i, j - 1], Y2[i, j - 1])
    return (1/beta) * np.log(max(1, X2[n_x, n_y], Y2[n_x, n_y], M[n_x, n_y]))

#-----------------------------------------------------------------------------------
#                                      String Kernel -- very slow
#-----------------------------------------------------------------------------------

def B_k(lamda, k, x, y):
    if k == 0:
        return 1
    if len(x) < k or len(y) < k:
        return 0
    sub_x, sub_y = x[:-1], y[:-1]
    return ( lamda * B_k(lamda, k, sub_x, y)
             + lamda * B_k(lamda, k, x, sub_y)
             - (lamda**2) * B_k(lamda, k, sub_x, sub_y)
             + ((lamda**2) * B_k(lamda, k-1, sub_x, sub_y) if x[-1] == y[-1] else 0)
           )

def string_kernel(x, y, lamda, k):
    if k == 0:
        return 1
    if len(x) < k or len(y) < k:
        return 0
    sub_x = x[:-1]
    return ( string_kernel(lamda, k, sub_x, y)
         + (lamda**2) * sum(B_k(lamda, k-1, sub_x, y[:j]) for j in range(len(y)) if y[j]==x[-1])
           )
 
#-----------------------------------------------------------------------------------
#                                      END
#-----------------------------------------------------------------------------------
