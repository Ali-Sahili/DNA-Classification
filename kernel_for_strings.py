"""  https://github.com/timshenkao/StringKernelSVM/blob/master/stringSVM.py """

import numpy as np


def K1(n, s, t, decay_param):

    if n == 0:
        return 1
    elif min(len(s), len(t)) < n:
        return 0
    else:
        part_sum = 0
        for j in range(1, len(t)):
            if t[j] == s[-1]:
                part_sum += K1(n - 1, s[:-1], t[:j], decay_param) * (decay_param ** (len(t) - (j + 1) + 2))
        result = decay_param * K1(n, s[:-1], t, decay_param) + part_sum
        return result

def K(n, s, t, decay_param):

    if min(len(s), len(t)) < n:
        return 0
    else:
        part_sum = 0
        for j in range(1, len(t)):
            if t[j] == s[-1]:
                part_sum += K1(n - 1, s[:-1], t[:j], decay_param)
        result = K(n, s[:-1], t,decay_param) + decay_param ** 2 * part_sum
        return result



def gram_matrix_elem(s, t, sdkval1, sdkval2, subseq_length, decay_param):
    if s == t:
        return 1
    else:
        return K(subseq_length, s, t, decay_param) / (sdkval1 * sdkval2) ** 0.5

def string_kernel(X1, X2=[], subseq_length=3, decay_param=0.5):
    """ In our case we have fixed size where X1 and X2 has the same length if X2 exists """
    len_X1 = len(X1)
    len_X2 = len(X2)
    
    sim_docs_kernel_value = {}

    if len_X2 == 0:
        # numpy array of Gram matrix
        gram_matrix = np.zeros((len_X1, len_X1), dtype=np.float32)

        print("Starting...")
        # store K(s,s) values in dictionary to avoid recalculations
        for i in range(len_X1):
            print(i)
            sim_docs_kernel_value[i] = K(subseq_length, X1[i], X1[i],decay_param)
        
        print("Computing K...")    
        #calculate Gram matrix
        for i in range(len_X1):
            for j in range(i, len_X1):
                gram_matrix[i, j] = gram_matrix_elem(X1[i], X1[j], sim_docs_kernel_value[i],\
                           sim_docs_kernel_value[j], subseq_length, decay_param)
                gram_matrix[j, i] = gram_matrix[i, j]
                
    else:
        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)

        sim_docs_kernel_value[1] = {}
        sim_docs_kernel_value[2] = {}
        
        #store K(s,s) values in dictionary to avoid recalculations
        for i in range(len_X1):
            sim_docs_kernel_value[1][i] = K(subseq_length, X1[i], X1[i], decay_param)
        for i in range(len_X2):
            sim_docs_kernel_value[2][i] = K(subseq_length, X2[i], X2[i], decay_param)
        
        #calculate Gram matrix
        for i in range(len_X1):
            for j in range(len_X2):
                gram_matrix[i, j] = gram_matrix_elem(X1[i],X2[j],sim_docs_kernel_value[1][i],sim_docs_kernel_value[2][j], subseq_length, decay_param)
        
    return gram_matrix
