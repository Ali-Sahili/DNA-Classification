# DataChallenge - DNA-Classification

<p float="center">
  <img src="Image/DNA_image.jpg" width="900">
</p>

### Table of Content

- [Introduction](#Introduction)
- [Classifiers](#Classifiers)
- [Kernels on Numeric Data](#Kernels_on_Numeric_Data)
- [Kernels on sequences](#Kernels_on_sequences)
- [Deep learning](#Deep_learning)
- [Requirements](#Requirements)
- [Testing](#Testing)
- [References](#References)
- [Acknowledgments](#Acknowledgments)

### Introduction
Transcription factors (TFs) are regulatory proteins thatbind specific sequence motifs in the genome to activate orrepress transcription of target genes, thus genomes can beclassified as bound or not bound for a specific TF. The goalof this challenge is to implement some Machine Learningalgorithms for DNA sequence classification.

### Classifiers
we have implemented several classifiers to solve this problem.  First of all,  we built Kernel Ridge Regression (KRR) and Kernel Logistic Regression (KLR) as naive classifiers for such an application. In addition, we have implemented Support Vector Machine (SVM) algorithm with its three versions:  Large Margin SVM where we tried three different optimization methods (SLQP - very slow, L-BFGS and CVX library), C-SVM also using CVX library and L-BFGS and 2-SVM using the CVX library.

### Kernels on Numeric Data
Fisrt approach is to work on a numeric version of sequences where each sequence (of size l=10) is represented as a vector of 4xl dimensions using one-got encoding (with A=(1, 0, 0, 0), C=(0, 1, 0, 0), G=(0, 0, 1, 0), T=(0, 0, 0, 1)). Then, they are clustered into 100 clusters using Kmeans and each subsequence is assigned to a label i and is represented by a binary vector whose coefficients are equal to 0 except the ith one, which is equal to 1. Finally, for each sequence, we compute the average of the representations of all its subsequences to obtain the feature vector of this sequence. You can find these data in data folder (e.g. Xtr1_mat100, Xte1_mat100).
Working on the numeric data, we tried five main kernels:linear, RBF, quadratic , polynomial and sigmoid[6].

### Kernels on sequences
Finding a convinient kernel for such an application is popular problem widely studied over years. In this project, we have implemented some of the famous kernels for DNA or protein classification: Spectrum kernel[1], Mismatch Kernel, weighted substring kernel w/o shifts[2], Local Alignment Kernels[3].
To improve results, we tried a Multi-Kernel Learning (MKL) approach to combining or mixing best kernels under specific weights.

### Deep learning
Due to the ability of deep learning methods to extract robust and useful features rather than its huge performance over the last few years, we tried to implement one CNN-based method which also relied on the concept of kernels for sequences. We have implemented the paper [7] which consists of introducing a Convolutional Kernel Network (CKN), a little bit different from the standard CNN.

### Requirements
The experiments were performed using Python 3.8.5 with the following Python packages:
* [numpy](http://www.numpy.org/) == 1.18.5
* [pandas](https://pandas.pydata.org/) == 0.25.3
* [tqdm](https://tqdm.github.io/) == 4.30.0
* [cvxopt](https://cvxopt.org/) == 1.2.5
* [scipy](https://www.scipy.org/) == 1.3.3

For Deep learning method, you need to install some additional libraries:
* [torch](https://pytorch.org/) == 1.5.1
* [sklearn](https://scikit-learn.org/stable/) == 0.22.2

### Testing
To test results using numeric data, choose the corresponding parameters for each method and kernel:
```
python main_numeric.py --path data/ --out-path results/
                       --method CSVM --C 5. --iters 100
                       --solver CVX --PCA False
                       --kernel rbf --gamma 0.01
                       --cross-val True --kfolds 5
                       --trw True  --normalize False 
                       --save True
```

To test results using sequences directly, choose also the corresponding parameters for each method and kernel:
```
python main_sequence.py --path data/ --out-path results/
                        --method CSVM --C 5. --iters 100
                        --solver CVX --PCA False
                        --kernel spectrum --nplets 10
                        --trw True  --normalize False
                        --save True
```

In addition, to work with deep learning method, choose your parameter:
```
python Deep_method/main.py --path ../data/ --kernel_func exp --kernel_args 0.2
                           --alpha 1e-4 --penalty l1 --noise 0.01
                           --lr 0.01 --epochs 500
                           --n_motifs 128 256 128
                           --len_motifs 8 6 6
                           --stride 1 1 1
                           --use_cuda False
```

To reproduce our best results, put into your terminal:
```
python start.py
```

### References
<a id="1">[1]</a> 
[The spectrum kernel: A string kernel for SVM protein classification.](https://www.ics.uci.edu/~welling/teatimetalks/kernelclub04/spectrum.pdf)
  
<a id="1">[2]</a> 
[Efficient Sequence Modeling with String Kernels](https://mstrazar.github.io/tutorial/python/machine-learning/2018/08/31/string-kernels.html)

<a id="1">[3]</a> 
[Local Alignment Lecture](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/local.pdf)

<a id="1">[4]</a> 
[The Kernel Cookbook: Advice on Covariance functions by David Duvenaud](http://www.cs.toronto.edu/~duvenaud/cookbook/index.html)

<a id="1">[5]</a>
[Theoretical properties and implementation of the one-sided mean kernel for time series.](https://www.sciencedirect.com/science/article/pii/S0925231215003665)

<a id="1">[6]</a>
[Support Vector Machines and Kernels for Computational Biology](http://www.raetschlab.org/lectures/ismb09tutorial/handout.pdf)

<a id="1">[7]</a> 
[Biological sequence modeling with convolutional kernel networks](https://hal.inria.fr/hal-01632912v3/document)

### Acknowledgments
- StringKernelSVM by [Tim_Shenkao](https://github.com/timshenkao/StringKernelSVM/blob/master/stringSVM.py)
- Kernel DNA Classification by [Shahine](https://github.com/shahineb/kernel_dna_classification)
- CKN model implementation by [CHEN Dexiong](https://gitlab.inria.fr/dchen/CKN-seq)
