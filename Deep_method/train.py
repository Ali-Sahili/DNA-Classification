import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import copy

def train_model(  model, train_loader, val_loader, criterion, optimizer, lr_scheduler=None, 
                  epochs=100, use_cuda=False):
        
    print("Initializing CKN layers")
    model.unsup_train_ckn(train_loader, use_cuda=use_cuda)

    print("Finished, elapsed time.")
    print("===================================================================")
    
    phases = ['train']
    data_loader = {'train': train_loader}
    if val_loader is not None:
        phases.append('val')
        data_loader['val'] = val_loader

    epoch_loss = None
    best_loss = float('inf')
    best_acc = 0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)
        model.train(False)
        model.unsup_train_classifier(train_loader, criterion, use_cuda=use_cuda)
        
        for phase in phases:
            if phase == 'train':
                if lr_scheduler is not None:
                    if isinstance(lr_scheduler, ReduceLROnPlateau):
                        if epoch_loss is not None:
                            lr_scheduler.step(epoch_loss)
                    else:
                        lr_scheduler.step()
                    print("current LR: {}".format(optimizer.param_groups[0]['lr']))
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data, target, *_ in data_loader[phase]:
                size = data.size(0)
                target = target.float()
                if use_cuda:
                    data = data.cuda()
                    target = target.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                if phase == 'val':
                    with torch.no_grad():
                        output = model(data).view(-1)
                        pred = (output.data > 0).float()
                        loss = criterion(output, target)
                else:
                    output = model(data).view(-1)
                    pred = (output > 0).float()
                    loss = criterion(output, target)
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    model.normalize_()

                # statistics
                running_loss += loss.item() * size
                running_corrects += torch.sum(pred == target.data).item()

            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_corrects / len(data_loader[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if (phase == 'val') and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_weights = copy.deepcopy(model.state_dict())

        print()

    print('Finish at epoch: {}'.format(epoch + 1))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_weights)

    return model
