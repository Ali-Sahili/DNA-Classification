import numpy as np
import pandas as pd

def read_data(path):
  """ read data from .csv file """
  X = np.genfromtxt(path, delimiter=' ')
  return X

def read_labels(path):
  """ read labels by changing them from 0/1 to -1/1   """
  Y = pd.read_csv(path)['Bound']
  Y = np.where(Y==0, -1, Y)
  return Y
  
def read_sequence(path, type_="numpy"):
  """ read sequence data """
  sequence = pd.read_csv(path)['seq']
  if type_=="numpy":
      return sequence.to_numpy()
  return sequence # pandas

# Sigmoid function
def Sigmoid(v):
  return 1/(1+np.exp(-v))


# Score    
def score(pred, labels):
  return np.mean(pred==labels)

# Try an existing implementation from sklearn to test our results
def KRR_sklearn(X, X_test, y, lam=0.01):
  from sklearn import kernel_ridge as kr
  from sklearn.metrics import pairwise
  n = X.shape[0]
  model = kr.KernelRidge(alpha = lam*n, kernel = 'rbf')
  model.fit(X, y)
  pred = model.predict(X)
  pred_test = model.predict(X_test)
  return pred, pred_test

# Save results in the right format 
def save_results(pred, out_path="", filename="results"):
  import pandas as pd
  data_id = np.arange(pred.shape[0])
  data_val = pred
  data_val = np.where(data_val==-1, 0, 1)
  
  df = pd.DataFrame({"Id" : data_id, "Bound" : data_val})
  df.to_csv(out_path + filename + ".csv", index=False)
   
def save_results_(pred_0, pred_1, pred_2, out_path="", filename="results"):
  import pandas as pd
  data_id = np.arange(pred_0.shape[0] + pred_1.shape[0] + pred_2.shape[0])
  data_val = np.concatenate([pred_0, pred_1, pred_2])
  data_val = np.where(data_val==-1, 0, 1)

  df = pd.DataFrame({"Id" : data_id, "Bound" : data_val})
  df.to_csv(out_path + filename + ".csv", index=False)

# Cross Validation
def cross_validation(algo, X, y, kfolds=5):
    n = X.shape[0]
    p = n // kfolds
    
    for i in range(kfolds):
        # randomly choose p samples for validation set
        idx_val = np.random.choice(n, p)
        idx_train = np.setdiff1d(np.arange(n), idx_val)
    
        # Prepare training and validation sets
        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]
    
        # training
        algo.fit(X_train, y_train)
        
        # Compute score on training set
        pred_tr = algo.predict(X_train)
        print("Fold "+str(i+1)+" - score on training set:   ", score(pred_tr, y_train))

        # Compute score on validation set
        pred_val = algo.predict(X_val)
        print("Fold "+str(i+1)+" - score on validation set: ", score(pred_val, y_val))

# train algorithm on the training set and predict labels of the testing set
def fit_and_predict(algo, X, y, X_test, return_score=False, ratio=0.2, verbose=False, suffix='test', save=False):
    # ratio: default 20% validation set and 80 % training set
    n = X.shape[0]
    p = int(ratio*n) # 20% for validation
    
    # randomly choose p samples for validation set
    idx_val = np.random.choice(n, p)
    idx_train = np.setdiff1d(np.arange(n), idx_val)
    
    # Prepare training and validation sets
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]

    try:
        algo.fit(X_train, y_train, suffix=suffix, save=save)
    except:
        algo.fit(X_train, y_train)
    #print(algo.alpha)
    
    # Compute score on training set
    pred = algo.predict(X_train)
    if verbose: print("score on training set:   ", score(pred, y_train))

    # Compute score on validation set
    pred_val = algo.predict(X_val)
    sc_val = score(pred_val, y_val)
    if verbose: 
      print("score on validation set: ", sc_val)
      print()
    
    pred_test = algo.predict(X_test)
    
    if return_score: return pred_test, sc_val
    return pred_test




# Normalizing the Gram matrix
def normalize_gram(K):
    if K[0, 0] == 1:
        print('Kernel already normalized!!')
    else:
        n = K.shape[0]
        diag = np.sqrt(np.diag(K))
        for i in range(n):
            for j in range(i+1, n):
                K[i, j] /= (diag[i] * diag[j])
                K[j, i] = K[i, j]
        np.fill_diagonal(K, np.ones(n))
    return K

# convert sequence to numeric
def seq_to_num(x):
  def func(x): 
      x = np.char.replace(x, 'A', '1')
      x = np.char.replace(x, 'C', '2')
      x = np.char.replace(x, 'G', '3')
      x = np.char.replace(x, 'T', '4')
      return x
  
  res = list(str(func(x)))
  #print(res)
  res = np.array(res).astype(int) 
  #print(res)
  return res

# showing the evolution of similarity between sequences  
def plot_gram_matrix(K, n=100):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    _ = plt.pcolor(K[:n, :n])
    _ = plt.xlabel("Sequence evolution $\\rightarrow$")
    _ = plt.ylabel("Sequence evolution $\\rightarrow$")
