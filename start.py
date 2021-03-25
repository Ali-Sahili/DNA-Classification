import numpy as np
import pandas as pd
from KSVM import C_SVM
from utils import read_sequence, read_labels, fit_and_predict, save_results_


#---------------------------------------------------------------------------
#                           Parameters settings
#---------------------------------------------------------------------------
path = "data"
out_path = "./"

method = "CSVM"
C = 1.

kernel = "mismatch"
nplets = 9

val_ratio = 0.2


#---------------------------------------------------------------------------
#                    Prepare Data (train, test, labels)
#---------------------------------------------------------------------------
# Read training and testing datasets and labels as numeric
X_seq_0 = read_sequence(path + "/Xtr0.csv", type_="numpy")
X_seq_test_0 = read_sequence(path + "/Xte0.csv", type_="numpy")
Y_0 = read_labels(path + "/Ytr0.csv")
print("size of dataset0: ", X_seq_0.shape, X_seq_test_0.shape, Y_0.shape)

X_seq_1 = read_sequence(path + "/Xtr1.csv", type_="numpy")
X_seq_test_1 = read_sequence(path + "/Xte1.csv", type_="numpy")
Y_1 = read_labels(path + "/Ytr1.csv")
print("size of dataset1: ", X_seq_1.shape, X_seq_test_1.shape, Y_1.shape)

X_seq_2 = read_sequence(path + "/Xtr2.csv", type_="numpy")
X_seq_test_2 = read_sequence(path + "/Xte2.csv", type_="numpy")
Y_2 = read_labels(path + "/Ytr2.csv")
print("size of dataset0: ", X_seq_2.shape, X_seq_test_2.shape, Y_2.shape)


#---------------------------------------------------------------------------
#                             Algotithm Settings
#---------------------------------------------------------------------------
# Prepare the chosen method                                                     
print('**************  CSVM  *****************')
algo = C_SVM( kernel=kernel, C=C, solver="BFGS", normalize=False, nplets=nplets)

print("*****  mismatch k=" + str(nplets) + "*****")

#---------------------------------------------------------------------------
#                          Training and Prediction phase 
#---------------------------------------------------------------------------
for i in range(9):
    pred_test_0 = fit_and_predict(algo=algo, X=X_seq_0, y=Y_0, X_test=X_seq_test_0, verbose=True,
                                  save=False, ratio=val_ratio)

    pred_test_1 = fit_and_predict(algo=algo, X=X_seq_1, y=Y_1, X_test=X_seq_test_1, verbose=True,
                                  save=False, ratio=val_ratio)

    pred_test_2 = fit_and_predict(algo=algo, X=X_seq_2, y=Y_2, X_test=X_seq_test_2, verbose=True,
                                 save=False, ratio=val_ratio)

    # Save results in the right format
    print("Saving...")
    save_results_(pred_test_0, pred_test_1, pred_test_2, out_path, 
                   filename= str(i)"_results_mismatch")




#---------------------------------------------------------------------------
#                                Ensembling
#---------------------------------------------------------------------------
csv_1 = '0_results_mismatch.csv'
csv_2 = '1_results_mismatch.csv'
csv_3 = '2_results_mismatch.csv'
csv_4 = '3_results_mismatch.csv'
csv_5 = '4_results_mismatch.csv'
csv_6 = '5_results_mismatch.csv'
csv_7 = '6_results_mismatch.csv'
csv_8 = '7_results_mismatch.csv'
csv_9 = '8_results_mismatch.csv'

csv1 = pd.read_csv(csv_1)
csv2 = pd.read_csv(csv_2)
csv3 = pd.read_csv(csv_3)
csv4 = pd.read_csv(csv_4)
csv5 = pd.read_csv(csv_5)
csv6 = pd.read_csv(csv_6)
csv7 = pd.read_csv(csv_7)
csv8 = pd.read_csv(csv_8)
csv9 = pd.read_csv(csv_9)


combined = csv1.copy()

combined['Bound'] = csv1['Bound'] + csv2['Bound'] + csv3['Bound'] + csv4['Bound'] + csv5['Bound'] + csv6['Bound'] + csv7['Bound'] + csv8['Bound'] + csv9['Bound']

for i in range(len(combined['Bound'])):
   combined['Bound'][i] = int( combined['Bound'][i] >= 5)
 
combined.to_csv('Yte.csv', index=False)


#---------------------------------------------------------------------------
#                                    END
#---------------------------------------------------------------------------
