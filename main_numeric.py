import argparse
import numpy as np

from utils import read_data, read_labels
from utils import fit_and_predict, cross_validation, save_results, save_results_


#---------------------------------------------------------------------------
#                          Parameters Settings
#---------------------------------------------------------------------------
# 
parser = argparse.ArgumentParser(description='Data Callenge training script')
parser.add_argument('--path', type=str, default='data', 
                    metavar='D',help="folder where data is located.")
parser.add_argument('--method', type=str, default='CSVM', metavar='D',
                    help="Choose method: CSVM/2SVM/SVM/KRR/KLR/KRR_sklearn")
parser.add_argument('--kernel', type=str, default='linear', metavar='D',
                    help="type of kernel")
parser.add_argument('--iters', type=int, default=100, metavar='N',
                    help='number of iterations for KLR (default: 100)')                    
parser.add_argument('--lam', type=float, default=0.76, metavar='lambda',
                    help='regularization factor (default: 0.76)')
parser.add_argument('--eps', type=float, default=0.001, metavar='tolerence',
                    help='tolerence (default: 0.001)')                   
parser.add_argument('--gamma', type=float, default=0.01, metavar='gamma',
                    help='parameter for RBF kernel (default: 0.01)')                     
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the polynomial kernel (default: 2)')  
parser.add_argument('--C', type=float, default=10, metavar='N',
                    help='regularization term of CSVM (default: 10)')
parser.add_argument('--solver', type=str, default='BFGS', metavar='D',
       help="solver for the optimization problem BFGS/CVX for CSVM or  BFGS/SLSPQ for KSVM ")
parser.add_argument('--PCA', type=bool, default=False, metavar='N',
                    help='Apply PCA before classification (default: False)')  
parser.add_argument('--nc', type=int, default=80, metavar='N',
                    help='number of components of PCA (default: 80)') 
parser.add_argument('--save', type=bool, default=True, metavar='N',
                    help='Save results (default: True)')
parser.add_argument('--cross-val', type=bool, default=False, metavar='N',
                    help='Apply cross validation (default: False)') 
parser.add_argument('--kfolds', type=int, default=5, metavar='N',
                    help='number of folds in cross validation (default: 5)')  
parser.add_argument('--trw', type=bool, default=False, metavar='N',
                    help='Train the whole dataset at once (default: False)')
parser.add_argument('--out-path', type=str, default='./', 
                    metavar='D',help="folder where output is located.")                     
parser.add_argument('--normalize', type=bool, default=False, metavar='N',
                    help='normalizing the gram matrix (default: False)')
parser.add_argument('--nplets', type=int, default=3, metavar='N',
                    help='number of n-plets for "GACT" combination (default: 3)') 
parser.add_argument('--shift', type=int, default=2, metavar='N',
                    help='Shift value for Weighted kernel with shifts (default: 2)') 
parser.add_argument('--gap', type=int, default=2, metavar='N',
                    help='gap value for gappy kernel (default: 2)') 
args = parser.parse_args()


#---------------------------------------------------------------------------
#                    Prepare Data (train, test, labels)
#---------------------------------------------------------------------------
# Read training and testing datasets and labels as numeric
X_0 = read_data(args.path + "/Xtr0_mat100.csv")
X_test_0 = read_data(args.path + "/Xte0_mat100.csv")
Y_0 = read_labels(args.path + "/Ytr0.csv")
print("size of dataset0: ", X_0.shape, X_test_0.shape, Y_0.shape)

X_1 = read_data(args.path + "/Xtr1_mat100.csv")
X_test_1 = read_data(args.path + "/Xte1_mat100.csv")
Y_1 = read_labels(args.path + "/Ytr1.csv")
print("size of dataset1: ", X_1.shape, X_test_1.shape, Y_1.shape)

X_2 = read_data(args.path + "/Xtr2_mat100.csv")
X_test_2 = read_data(args.path + "/Xte2_mat100.csv")
Y_2 = read_labels(args.path + "/Ytr2.csv")
print("size of dataset2: ", X_2.shape, X_test_2.shape, Y_2.shape)

if args.trw:
    X = np.concatenate([X_0,X_1,X_2])
    Y = np.concatenate([Y_0,Y_1,Y_2])
    X_test = np.concatenate([X_test_0, X_test_1, X_test_2])

#---------------------------------------------------------------------------
#                    Principle Component Analysis (PCA)
#---------------------------------------------------------------------------
# Apply PCA before classification
if args.PCA:
    from KPCA import rbf_kernel_pca
    
    if args.trw:
        X = rbf_kernel_pca(X, kernel=args.kernel, gamma=args.gamma, 
                             degree=args.degree, n_components=args.nc)
        X_test = rbf_kernel_pca(X_test, kernel=args.kernel, gamma=args.gamma, 
                                        degree=args.degree, n_components=args.nc)
    else:
        X_0 = rbf_kernel_pca(X_0, kernel=args.kernel, gamma=args.gamma, 
                                  degree=args.degree, n_components=args.nc)
        X_1 = rbf_kernel_pca(X_1, kernel=args.kernel, gamma=args.gamma, 
                                  degree=args.degree, n_components=args.nc)
        X_2 = rbf_kernel_pca(X_2, kernel=args.kernel, gamma=args.gamma, 
                                  degree=args.degree, n_components=args.nc)

        X_test_0 = rbf_kernel_pca(X_test_0, kernel=args.kernel, gamma=args.gamma, 
                                            degree=args.degree, n_components=args.nc)
        X_test_1 = rbf_kernel_pca(X_test_1, kernel=args.kernel, gamma=args.gamma, 
                                            degree=args.degree, n_components=args.nc) 
        X_test_2 = rbf_kernel_pca(X_test_2, kernel=args.kernel, gamma=args.gamma, 
                                            degree=args.degree, n_components=args.nc)


#---------------------------------------------------------------------------
#                             Algotithm Settings
#---------------------------------------------------------------------------
# Prepare the chosen method                                                     
if args.method == "CSVM":
    print('**************  CSVM  *****************')
    from KSVM import C_SVM
    algo = C_SVM( kernel=args.kernel, C=args.C, solver=args.solver,
                  gamma=args.gamma, degree=args.degree, normalize=args.normalize,
                  gap=args.gap, shift=args.shift, nplets=args.nplets)

elif args.method == "2SVM":
    print('**************  2-SVM  *****************')
    from KSVM import Two_SVM
    algo = Two_SVM(kernel=args.kernel, lam=args.lam, gamma=args.gamma, 
                     degree=args.degree, normalize=args.normalize,
                  gap=args.gap, shift=args.shift, nplets=args.nplets)
    
elif args.method == "SVM": # Max margin classifier
    print('**************  SVM  *****************')
    from KSVM import LargeMarginClassifier
    algo = LargeMarginClassifier(kernel=args.kernel, solver=args.solver, lam=args.lam,
                                gamma=args.gamma, degree=args.degree, normalize=args.normalize,
                                 gap=args.gap, shift=args.shift, nplets=args.nplets)

elif args.method == "KRR":
    print('**************  KRR  *****************')
    from KR import KRR
    algo = KRR(kernel=args.kernel, lam=args.lam, gamma=args.gamma, degree=args.degree)

elif args.method == "KLR":
    print('**************  KLR  *****************')
    from KR import KLR
    algo = KLR( kernel=args.kernel, lam=args.lam, eps=args.eps, 
               gamma=args.gamma, nb_iters=args.iters, degree = args.degree)


#elif args.method == "KRR_sklearn":
 #   from utils import KRR_sklearn
 #   print('**********  KRR sklearn  *************')
 #   pred_0, pred_test_0 = KRR_sklearn(X_0, X_test_0, Y_0, lam=args.lam)
 #   #print(np.sum(pred_0==1))
 #   sc = score(np.sign(pred_0), Y_0)
 #   print("score 0 on training set: ", sc)
    
 #   pred_1, pred_test_1 = KRR_sklearn(X_1, X_test_1, Y_1, lam=args.lam)
 #   #print(np.sum(pred_1==1))
 #   sc = score(np.sign(pred_1), Y_1)
 #   print("score 1 on training set: ", sc)
    
 #   pred_2, pred_test_2 = KRR_sklearn(X_2, X_test_2, Y_2, lam=args.lam)
 #   #print(np.sum(pred_2==1))
 #   sc = score(np.sign(pred_2), Y_2)
 #   print("score 2 on training set: ", sc)
    
else:
    raise NotImplemented

 
#---------------------------------------------------------------------------
#                    Training and Prediction phase 
#---------------------------------------------------------------------------
if args.trw:
    # train and predict of the whole dataset
    pred_test = fit_and_predict( algo=algo, X=X, y=Y, X_test=X_test, verbose=True)
else:
    # Train and predict for each dataset apart
    pred_test_0 = fit_and_predict(algo=algo, X=X_0, y=Y_0, X_test=X_test_0, verbose=True)
    pred_test_1 = fit_and_predict(algo=algo, X=X_1, y=Y_1, X_test=X_test_1, verbose=True)
    pred_test_2 = fit_and_predict(algo=algo, X=X_2, y=Y_2, X_test=X_test_2, verbose=True)


#---------------------------------------------------------------------------
#                            Cross Validation
#---------------------------------------------------------------------------
# Cross validation --> useful to choose algo's parameters
if args.cross_val:
    cross_validation( algo=algo, 
                      X=np.concatenate([X_0,X_1,X_2]), 
                      y=np.concatenate([Y_0, Y_1, Y_2]), 
                      kfolds=args.kfolds)


#---------------------------------------------------------------------------
#                               Save Results
#---------------------------------------------------------------------------
# Save results in the right format
if args.save:
    print("Saving...")
    if args.trw:
        save_results(pred_test, args.out_path)
    else:
        save_results_(pred_test_0, pred_test_1, pred_test_2, args.out_path)
#---------------------------------------------------------------------------
#                                    END
#---------------------------------------------------------------------------
