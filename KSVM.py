import numpy as np
from kernels import RBF_Gram_Matrix
from scipy import optimize

import cvxopt
import cvxopt.solvers


# Kernel SVM - Large margin classifier
class LargeMarginClassifier:
    
    def __init__(self, lam=1., kernel="rbf", solver = "BFGS", degree=3, 
                       gamma=0.01, eps=1e-3, normalize=False,
                        gap=2, nplets=3, shift=2):
        
        self.degree = degree
        self.tolerence = eps
        self.gamma = gamma
        self.kernel = kernel
        self.lamda = lam
        self.solver = solver
        self.normalize = normalize
        self.gap = gap
        self.nplets = nplets
        self.shift = shift
        
    def fit(self, X, y):
        n = X.shape[0]
        self.X = X
        
        # Gram matrix
        K = RBF_Gram_Matrix(X, [], self.kernel, self.gamma, self.degree, self.shift, 
                                  self.normalize, self.gap, self.nplets)

        # Lagrange dual problem
        def Ld0(K, y, mu):
            A = np.dot(np.diag(y), np.dot(K, np.diag(y)))
            return np.sum(mu) - (1./(4*self.lamda)) * np.dot(mu.T, np.dot(A, mu))

        # Partial derivate of Ld on mu
        def Ld0dmu(K, y, mu):
            A = np.dot(np.diag(y), np.dot(K, np.diag(y)))
            return np.ones_like(mu) - (1/(2*self.lamda)) * np.dot(A, mu)


        if self.solver == "SLSQP":
            # Constraints on mu of the shape :
            # -  1/n - mu  >= 0 --> d - C*mu  = 0
            # -  mu >= 0        --> b - A*mu  = 0
            A = -np.eye(n)
            b = np.zeros(n)
            C = np.eye(n)
            d = np.ones(n)*(1/n)
            constraints = (
                  {'type': 'ineq', 'fun': lambda mu: d - np.dot(C,mu), 'jac': lambda mu: -C},
                  {'type': 'ineq', 'fun': lambda mu: b - np.dot(A,mu), 'jac': lambda mu: -A})

            # Initial condition
            x0 = np.zeros(n)

            # Options
            opt = {'disp':True}

            # Maximize by minimizing the opposite
            optRes = optimize.minimize( fun=lambda mu: -Ld0(K, y, mu),
                                        x0=x0, 
                                        method='SLSQP', 
                                        jac=lambda mu: -Ld0dmu(K, y, mu), 
                                        constraints=constraints,
                                        options=opt)

            self.mu = optRes.x
            self.alpha = (1/(2*self.lamda)) * np.dot(np.diag(y), self.mu)

        elif self.solver == "BFGS":
            # initialization
            mu0 = np.random.randn(n)
            # Gradient descent
            bounds_down = [0 for i in range(n)]
            bounds_up = [1/n for i in range(n)]
            bounds = [[bounds_down[i], bounds_up[i]] for i in range(n)]
            res = optimize.fmin_l_bfgs_b(lambda mu: -Ld0(K, y, mu), mu0, 
                                         fprime=lambda mu: -Ld0dmu(K, y, mu), bounds=bounds)
            self.mu = res[0]
            self.alpha = (1/(2*self.lamda)) * np.dot(np.diag(y), self.mu)

        elif self.solver == "CVX":
            cvxopt.solvers.options['show_progress'] = False

            q = -cvxopt.matrix(y, (n, 1), tc='d')
            h = cvxopt.matrix(np.concatenate([np.ones(n)/(2*self.lamda*n), np.zeros(n)]).reshape((2*n, 1)))
            P = cvxopt.matrix(K)
            Gtop = cvxopt.spmatrix(y, range(n), range(n))
            G = cvxopt.sparse([Gtop, -Gtop])

            sol = cvxopt.solvers.qp(P, q, G, h)
            self.alpha = np.ravel(sol['x'])

        else: raise NotImplemented

    def predict(self, x_test):
        K = RBF_Gram_Matrix(x_test, self.X, self.kernel, self.gamma, self.degree, self.shift, 
                                  self.normalize, self.gap, self.nplets)
        return np.sign(np.dot(K, self.alpha))

    def score(self, pred, labels):
        return np.mean(pred==labels)
        

# Kernel SVM - C-SVM equivalent to dual Max margin classifier
class C_SVM:
    def __init__(self, kernel="rbf", C=10, solver='BFGS', gamma=0.01, degree=3, normalize=False,
                         shift=2, nplets=3, gap=2):

        self.C = C
        self.gamma = gamma
        self.solver = solver
        self.degree = degree
        self.kernel = kernel
        self.normalize = normalize
        self.gap = gap
        self.nplets = nplets
        self.shift = shift

    def loss(self, alpha):
        return -(2 * np.dot(alpha, self.y) - np.dot(alpha.T, np.dot(self.K, alpha)))

    def jac(self, alpha):
        return -(2 * self.y - 2*np.dot(self.K, alpha))

    def fit(self, X, y):
        
        self.X = X
        self.y = y

        # Gram Matrix
        self.K = RBF_Gram_Matrix(X, [], self.kernel, self.gamma, self.degree, self.shift, 
                                        self.normalize, self.gap, self.nplets)
        n = self.K.shape[0]
        
        if self.solver == 'BFGS':
            # initialization
            alpha0 = np.random.randn(n)
            # Gradient descent
            # 0 <= alpha*y <= C
            bounds_down = [-self.C if self.y[i] <= 0 else 0 for i in range(n)]
            bounds_up = [+self.C if self.y[i] >= 0 else 0 for i in range(n)]
            bounds = [[bounds_down[i], bounds_up[i]] for i in range(n)]
            res = optimize.fmin_l_bfgs_b(self.loss, alpha0, fprime=self.jac, bounds=bounds)
            self.alpha = res[0]
            
        elif self.solver == 'CVX':
            cvxopt.solvers.options['show_progress'] = False
            r, o, z = np.arange(n), np.ones(n), np.zeros(n)
            P = cvxopt.matrix(self.K.astype(float), tc='d')
            q = cvxopt.matrix(-self.y, tc='d')
            G = cvxopt.spmatrix(np.r_[self.y, -self.y], np.r_[r, r + n], np.r_[r, r], tc='d')
            h = cvxopt.matrix(np.r_[o * self.C, z], tc='d')
            cvxopt.solvers.options['show_progress'] = False
            sol = cvxopt.solvers.qp(P, q, G, h)
            self.alpha = np.ravel(sol['x'])

    def predict(self, x_test):
        K = RBF_Gram_Matrix(x_test, self.X, self.kernel, self.gamma, self.degree, self.shift, 
                                            self.normalize, self.gap, self.nplets)
        return np.sign(np.dot(K, self.alpha))

    def score(self, pred, labels):
        return np.mean(pred==labels)

# Kernel SVM - 2-SVM with square hinge loss
class Two_SVM:
    def __init__(self, kernel="rbf", lam=0.01, gamma=0.01, degree=3, normalize=False,
                         gap=2, shift=2, nplets=3):

        self.lamda = lam
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel
        self.normalize = normalize
        self.gap = gap
        self.nplets = nplets
        self.shift = shift

    def fit(self, X, y):
        cvxopt.solvers.options['show_progress'] = False
            
        self.X = X
        n = X.shape[0]
            
        # Gram Matrix
        self.K = RBF_Gram_Matrix(X, [], self.kernel, self.gamma, self.degree, self.shift, 
                                        self.normalize, self.gap, self.nplets)
            
        P = cvxopt.matrix(2 * (self.K + n * self.lamda * np.identity(n)), tc="d")
        q = cvxopt.matrix(- 2 * y, tc="d")
        G = cvxopt.matrix(np.diag(- y), tc="d")
        h = cvxopt.matrix(np.zeros((n, 1)), tc="d")
            
        solution = cvxopt.solvers.qp(P, q, G, h)
        self.alpha = np.ravel(solution["x"])

    def predict(self, x_test):
        K = RBF_Gram_Matrix(x_test, self.X, self.kernel, self.gamma, self.degree, self.shift, 
                                             self.normalize, self.gap, self.nplets)
        return np.sign(np.dot(K, self.alpha))

    def score(self, pred, labels):
        return np.mean(pred==labels)
