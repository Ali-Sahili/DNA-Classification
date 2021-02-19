import numpy as np
from kernels import RBF_Gram_Matrix
from utils import Sigmoid

# Kernel Logistic Regression method
class KLR:
    def __init__(self, kernel, lam=0.01, gamma=0.01, eps=1e-3, nb_iters=100, degree=2):
        super(KLR, self).__init__()

        self.lamda = lam
        self.gamma = gamma
        self.tolerence = eps
        self.degree = degree
        self.nb_iters = nb_iters
        self.kernel = kernel

    # kernelized version of Iteratively Reweighted Least-Square method
    def IRLS(self, K, y, alpha):
        m = np.dot(K, alpha)
        #P = -Sigmoid(-y*m)
        W = Sigmoid(m) * Sigmoid(-m)
        z = m + y/Sigmoid(-y*m)
        return W, z

    # Weighted Kernel Ridge Regression
    def solveWKRR(self, K, W, z):
        n = K.shape[0]
        Ws = np.diag(np.sqrt(W))
        tmp = np.dot(np.dot(Ws, K), Ws) + n * self.lamda * np.eye(n)
        A = np.dot(np.dot(Ws, np.linalg.inv(tmp)), Ws)
        alpha = np.dot(A, z)
        return alpha

    def fit(self, X, y):
        n = y.shape[0]
        self.X = X
        
        K = RBF_Gram_Matrix(X, [], self.kernel, self.gamma, self.degree)
        
        alpha_prev = np.zeros(n)
        for _ in range(self.nb_iters):
            W, z = self.IRLS(K, y, alpha_prev)
            alpha = self.solveWKRR(K, W, z)
            if (np.linalg.norm(alpha-alpha_prev)) < self.tolerence:
               break
            alpha_prev = alpha

        self.alpha = alpha_prev

    def predict(self, x_test):
        K = RBF_Gram_Matrix(x_test, self.X, self.kernel, self.gamma, self.degree)
        return np.sign(np.dot(K, self.alpha))

# Kernel Ridge Regression
class KRR:
    def __init__(self, kernel, lam=0.01, gamma=0.01, degree=2):
        super(KRR, self).__init__()
        
        self.lamda = lam
        self.gamma = gamma
        self.kernel = kernel
        self.degree = degree
        
    def fit(self, X, y):
        self.X = X
        n = X.shape[0]
        K = RBF_Gram_Matrix(X, [], self.kernel, self.gamma, self.degree)
        self.alpha = np.dot(np.linalg.inv(K + n*self.lamda * np.eye(n)), y.transpose())
    
    def predict(self, x_test):
        K = RBF_Gram_Matrix(x_test, self.X, self.kernel, self.gamma, self.degree)
        return np.sign(np.dot(K, self.alpha))
