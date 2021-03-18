import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import numpy as np

from models import CKN
from train import train_model
from prepare_data import load_data

#---------------------------------------------------------------------------
#                               Parameters Settings
#---------------------------------------------------------------------------
# 
parser = argparse.ArgumentParser(description='Data Callenge training script')
parser.add_argument('--path', type=str, default='../data/', 
                    metavar='D',help="folder where data is located.")
                    
parser.add_argument('--kernel_func', type=str, default='exp', metavar='D',
                    help="Choose kernel: exp or add_exp") 
parser.add_argument('--kernel_args', type=float, default=0.3, metavar='param',
                    help='kernel parameter (default: 0.3)')                  
parser.add_argument('--alpha', type=float, default=1e-6, metavar='alpha',
                    help='regularization factor (default: 1e-6)')  
parser.add_argument('--penalty', type=str, default='l1', metavar='D',
                    help="Choose regularization norm.")                   
parser.add_argument('--n_sampling_patches', type=int, default=250000, metavar='n_patches',
                    help='number of sampling patches (default: 250000)')                   
parser.add_argument('--noise', type=float, default=0.0, metavar='noise',
                    help='adding noise (default: 0.0)') 
parser.add_argument('--lr', type=float, default=0.01, metavar='lr',
                    help='learning rate (default: 0.01)')                    
parser.add_argument('--epochs', type=int, default=5, metavar='epochs',
                    help='number of epochs (default: 5)')
parser.add_argument('--in_channels', type=int, default=4, metavar='in_channels',
                    help='size of input channels (default: 4)')
parser.add_argument('--n_motifs', nargs="*", type=int, default=[128], metavar='n_motifs',
                    help='size of output channels (default: [128])')
parser.add_argument('--len_motifs', nargs="*", type=int, default=[12], metavar='len_motifs',
                    help='size of filters (default: [12])')
parser.add_argument('--stride', nargs="*", type=int, default=[1], metavar='strides',
                    help='size of stride operation (default: [1])')
parser.add_argument('--use_cuda', type=bool, default=False, metavar='use_cuda',
                    help='use cuda (default: False)')
                    
args = parser.parse_args()



#---------------------------------------------------------------------------
#                                Loading Data
#---------------------------------------------------------------------------
#
train_loader, val_loader = load_data(args.path+"Xtr0.csv", args.path+"Ytr0.csv", phase="train")
test_loader = load_data(args.path+"Xtr0.csv", args.path+"Ytr0.csv", phase="test")


#---------------------------------------------------------------------------
#                              Define the model
#---------------------------------------------------------------------------
#
model = CKN( args.in_channels, args.n_motifs, args.len_motifs, args.stride,
             args.kernel_func, args.kernel_args, args.alpha, args.penalty)
print("==========================   Model   ==============================")        
print(model)
print("===================================================================")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.ckn_model.parameters(), lr=args.lr)
patience = 6 if args.noise > 0 else 4
lr_scheduler = ReduceLROnPlateau( optimizer, factor=0.25, patience=patience, min_lr=1e-4)

#---------------------------------------------------------------------------
#                                Training
#---------------------------------------------------------------------------
#
model = train_model( model, train_loader, val_loader, criterion, optimizer, lr_scheduler,
                     args.epochs, args.use_cuda)

#---------------------------------------------------------------------------
#                                Testing
#---------------------------------------------------------------------------
#
y_pred, y_true = model.predict(test_loader, proba=True, use_cuda=args.use_cuda)
