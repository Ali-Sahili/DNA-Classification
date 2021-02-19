import argparse
import numpy as np
from KSVM import C_SVM, LargeMarginClassifier
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
parser.add_argument('--save', type=bool, default=True, metavar='N',
                    help='Save results (default: True)')
parser.add_argument('--out-path', type=str, default='./', 
                    metavar='D',help="folder where output is located.")                     
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

X = np.concatenate([X_0,X_1,X_2])
Y = np.concatenate([Y_0,Y_1,Y_2])
X_test = np.concatenate([X_test_0, X_test_1, X_test_2])


print('**************  CSVM  *****************')
# parameters = {'kernel':("lineal","rbf","quadratic"), 'C':range(5,50,5)}

#best_preds = None
for kernel in ["polynomial"]: #,["rbf", "linear", "shift_linear", "quadratic", "r_quadratic", "polynomial", "sigmoid", "exp_cos", "periodic", "l_periodic", "MKL"]:
    max_sc = 0
    best_kernel = ""
    best_gamma = -1
    best_C = -1
    best_norm = False
    for C in [5]: #range(5,50,5):
      for normalize in ["False"]: #["False","True"]:
        for degree in [3, 4, 5, 6]:
            algo = C_SVM(kernel=kernel, C=C, solver="CVX", degree=degree, normalize=normalize)
            #algo = LargeMarginClassifier(kernel=kernel, lam=C, solver="BFGS", degree=degree, normalize=normalize)
            pred_test,sc = fit_and_predict(algo=algo, X=X, y=Y, X_test=X_test, return_score=True)
            print("Normalize=", normalize, ", kernel=", kernel, ", C=", C,  ", degree=", degree,", score=", sc)
            if sc > max_sc:
                max_sc = sc
                best_kernel = kernel
                best_C = C
                best_gamma = degree
                best_norm = normalize
                #best_preds = pred_test

    print('==============================================================================')
    print("kernel=", best_kernel, ", C=", best_C,  ", gamma=", best_gamma,", score=", max_sc)
    print('==============================================================================')

if False:
    print("Saving...")                      
    save_results(best_preds, args.out_path+"results.csv")
