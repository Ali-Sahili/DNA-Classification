import argparse
from KR import KLR, KRR
from KPCA import rbf_kernel_pca
from kernels import *
import numpy as np
from utils import *
from KSVM import LargeMarginClassifier, C_SVM

# Training settings
parser = argparse.ArgumentParser(description='Data Callenge training script')
parser.add_argument('--path', type=str, default='data', 
                    metavar='D',help="folder where data is located.")
parser.add_argument('--method', type=str, default='CSVM', metavar='D',
                    help="Choose method: CSVM/SVM/KRR/KLR/KRR_sklearn")
parser.add_argument('--kernel', type=str, default='substring_mis_w', metavar='D',
                    help="type of kernel: linear-RBF-polynomial-quadratic or wdk-wdkws-")
parser.add_argument('--iters', type=int, default=100, metavar='N',
                    help='number of iterations for KLR (default: 100)')                    
parser.add_argument('--lam', type=float, default=0.01, metavar='lambda',
                    help='regularization factor (default: 0.01)')
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
parser.add_argument('--save', type=bool, default=True, metavar='N',
                    help='Save results (default: True)')
args = parser.parse_args()


#---------------------------------------------------------------------------
#                    Prepare Data (train, test, labels)
#---------------------------------------------------------------------------
# Read training and testing datasets and labels as numeric
X_seq_0 = read_sequence(args.path + "/Xtr0.csv", type_="numpy")
X_seq_test_0 = read_sequence(args.path + "/Xte0.csv", type_="numpy")
Y_0 = read_labels(args.path + "/Ytr0.csv")
print("size of dataset0: ", X_seq_0.shape, X_seq_test_0.shape, Y_0.shape)

X_seq_1 = read_sequence(args.path + "/Xtr1.csv", type_="numpy")
X_seq_test_1 = read_sequence(args.path + "/Xte1.csv", type_="numpy")
Y_1 = read_labels(args.path + "/Ytr1.csv")
print("size of dataset1: ", X_seq_1.shape, X_seq_test_1.shape, Y_1.shape)

X_seq_2 = read_sequence(args.path + "/Xtr2.csv", type_="numpy")
X_seq_test_2 = read_sequence(args.path + "/Xte2.csv", type_="numpy")
Y_2 = read_labels(args.path + "/Ytr2.csv")
print("size of dataset0: ", X_seq_2.shape, X_seq_test_2.shape, Y_2.shape)

if args.trw:
    X_seq = np.concatenate([X_seq_0,X_seq_1,X_seq_2])
    Y = np.concatenate([Y_0,Y_1,Y_2])
    X_seq_test = np.concatenate([X_seq_test_0, X_seq_test_1, X_seq_test_2])
    print("size of dataset: ", X_seq.shape, X_seq_test.shape, Y.shape)
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

else:
    raise NotImplemented


#---------------------------------------------------------------------------
#                              Kernels settings 
#---------------------------------------------------------------------------

#from kernel_for_strings import *
#gram_matrix = string_kernel(X_seq_0, subseq_length=50)

# for mean alignment kernel
#data = []
#for i in range(X_seq_0.shape[0]):
#    data.append(seq_to_num(X_seq_0[i]))
#np.array(data)

if args.kernel == "spectrum":
    print("**********************  spectrum k=" + str(args.nplets) + "  **********************")
elif args.kernel == "substring":
    print("*******  substring k=" + str(args.nplets) + ", delta=" + str(args.shift) + " *******")
elif args.kernel == "substring_mis":
    print("*****  substring_mis k=" + str(args.nplets) + ", delta=" + str(args.shift) + " *****")
elif args.kernel == "substring_mis_w":
    print("****  substring_mis_w k=" + str(args.nplets) + ", delta=" + str(args.shift) + " ****")
elif args.kernel == "wdk":
    print("***************  weighted degree kernel: d=" + str(args.degree) + " ****************")
elif args.kernel == "wdkws":
    print("  weighted degree with shifts: d=" + str(args.degree) + ",  shift=" + str(args.shift))
elif args.kernel == "gappy":
    print("*******  gappy kernel: k=" + str(args.nplets) + ", gap=" + str(args.gap) + " *******")
elif args.kernel == "MAK":
    print("**************  mean alignement kernel, sigma=" + str(args.gamma) + " **************")
elif args.kernel == "LAK":
    print("*****************************  local alignement kernel  ****************************")
elif args.kernel == "LAK_affine":
    print("****************************  affine alignement kernel   ***************************")
elif args.kernel == "needleman_wunsch":
    print("****************************    needleman wunsch kernel  ***************************")
elif args.kernel == "smith_waterman":
    print("*****************************   smith waterman kernel   ****************************")
elif args.kernel == "string":
    print("****   string kernel: k=" + str(args.nplets) + ", lamda=" + str(args.gamma) + " ****")
else:
    raise NotImplemented
 
#---------------------------------------------------------------------------
#                    Training and Prediction phase 
#---------------------------------------------------------------------------
if args.trw:
    # train and predict of the whole dataset
    pred_test = fit_and_predict( algo=algo, X=X_seq, y=Y, X_test=X_seq_test, verbose=True)
else:
    # Train and predict for each dataset apart
    pred_test_0 = fit_and_predict(algo=algo, X=X_seq_0, y=Y_0, X_test=X_seq_test_0, verbose=True)
    pred_test_1 = fit_and_predict(algo=algo, X=X_seq_1, y=Y_1, X_test=X_seq_test_1, verbose=True)
    pred_test_2 = fit_and_predict(algo=algo, X=X_seq_2, y=Y_2, X_test=X_seq_test_2, verbose=True)

#---------------------------------------------------------------------------
#                               Save Results
#---------------------------------------------------------------------------
# Save results in the right format
if args.save:
    print("Saving...")
    if args.trw:
        save_results(pred_test, args.out_path, filename="results_all_"+args.kernel)
    else:
        save_results_(pred_test_0, pred_test_1, pred_test_2, args.out_path, 
                                               filename="results_"+args.kernel)
#---------------------------------------------------------------------------
#                                    END
#---------------------------------------------------------------------------
