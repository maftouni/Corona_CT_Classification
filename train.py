import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from nets import *
import time, os, copy, argparse
import multiprocessing
from torchsummary import summary
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from torch import FloatTensor
from bayes_opt import BayesianOptimization

from gluoncv.gluoncv.model_zoo.residual_attentionnet import *
from residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

from models import UNet11_Attention, LinkNet34_Attention, UNet, UNet16_Attention, AlbuNet, DenseNet169, DenseNet121_Attention,DenseNet121, DenseNet121_reduced, previous_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, help="Training mode: finetue/transfer/scratch")
args= vars(ap.parse_args())

# Set training mode
train_mode=args["mode"]

# Set the train and validation directory paths
train_directory = 'data/training'
valid_directory = 'data/validation'
# Set the model save path
PATH="model.pth" 


# Number of epochs
num_epochs = 100
# Number of classes
num_classes = 2
# Number of workers
num_cpu = multiprocessing.cpu_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('num_cpu',num_cpu)
# Applying transforms to the data


image_transforms = { 
     'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
     ]),
     'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
     ])
    }
 
# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}
 
# Size of train and validation data
dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid'])
}



# Class names or target labels
#class_names = dataset['train'].classes
#print("Classes:", class_names)
    
# Print the train and validation data sizes
#print("Training-set size:",dataset_sizes['train'],
#      "\nValidation-set size:", dataset_sizes['valid'])

    


def load_model(lr,momentum,batchSize,decay):
# Set default device as gpu, if available
#device = torch.device("cpu")
    
    # Batch size
    bs = int(batchSize)
    
    if train_mode=='finetune':
        # Load a pretrained model - Resnet18
        print("\nLoading resnet18 for finetuning ...\n")
        model_ft = models.resnet50(pretrained=True)

        # Modify fc layers to match num_classes
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes )

    elif train_mode=='densenet':
        #model_ft = models.densenet121(pretrained=True)
        #print('model_ftmodel_ftmodel_ftmodel_ft',model_ft)
        # Modify fc layers to match num_classes
        #num_ftrs = model_ft.classifier.in_features
        #model_ft.classifier = nn.Linear(num_ftrs,num_classes )
        
        model_ft = DenseNet169(num_classes, pretrained=False)

    elif train_mode=='Attention':
        
        model_ft = ResidualAttentionModel(10)
        
        model = torch.load('model_resAttention.pth')
        model_ft.load_state_dict(model)
        
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes )
        
        
        
        #model_ft = DenseNet121_Attention(num_classes, pretrained=True)
    
    elif train_mode=='scratch':
        # Load a custom model - VGG11
        print("\nLoading VGG11 for training from scratch ...\n")
        model_ft = MyVGG11(in_ch=3,num_classes=11)

        # Set number of epochs to a higher value
        num_epochs=100

    elif train_mode=='transfer':
        # Load a pretrained model - MobilenetV2
        print("\nLoading mobilenetv2 as feature extractor ...\n")
        model_ft = models.mobilenet_v2(pretrained=True)    

    # Freeze all the required layers (i.e except last conv block and fc layers)
        for params in list(model_ft.parameters())[0:-5]:
            params.requires_grad = False

            # Modify fc layers to match num_classes
            num_ftrs=model_ft.classifier[-1].in_features
            model_ft.classifier=nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
            )    

    # Transfer the model to GPU
    #model_ft = model_ft.to(device)
    model_ft = nn.DataParallel(model_ft, device_ids=[0,1,2,3]).cuda()

    # Print model summary
    #print('Model Summary:-\n')
    #for num, (name, param) in enumerate(model_ft.named_parameters()):
    #     print(num, name, param.requires_grad )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer 
    #print('decaydecay',decay)
    
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    # Learning rate decay
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=35, gamma=0.1)
    return model_ft,criterion,  optimizer_ft,  exp_lr_scheduler, bs
#summary(model_ft, input_size=(3, 224, 224))
#print(model_ft)



# Model training routine 
print("\nTraining:-\n")
def train_model(model, criterion, optimizer, scheduler, bs,num_epochs=30):
    
    # Create iterators for data loading
    dataloaders = {
    'train':data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=False),
    'valid':data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=False)
}
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_f1 = 0.0

    # Tensorboard summary
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            predictions=FloatTensor()
            all_labels=FloatTensor()
            #torch.tensor([0.])
            # dataloaders,dataset_sizes = data_loader(train_directory,valid_directory)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                predictions = predictions.to(device, non_blocking=True)
                all_labels = all_labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print('outputshape',outputs, list(outputs.shape))
                    #print('labelslabels',labels, list(labels.shape))
                    _, preds = torch.max(outputs, 1)
                    #print(preds)
                    
                    loss = criterion(outputs, labels)
                    
                    #print('preds.float()',preds.float())
                    #print('predictions',predictions)
                    predictions=torch.cat([predictions,preds.float()])
                    all_labels=torch.cat([all_labels,labels.float()])
                    
                    #a = list(model.parameters())[0].clone()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        a0 = list(model.parameters())[0].clone()
                        loss.backward()
                        optimizer.step()
                        
                        #b = list(model.parameters())[0].clone()
                        #print('check training',torch.equal(a.data, b.data))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                
                
            if phase == 'train':
                scheduler.step()
                

            #print('all_labels',all_labels.tolist())
            #print('predictions',predictions.tolist())
            epoch_f1=f1_score(all_labels.tolist(), predictions.tolist())
            #print('epoch_f1',epoch_f1)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #print('{} Loss: {:.4f} Acc: {:.4f} f1: {:.4f}'.format(
            #   phase, epoch_loss, epoch_acc,epoch_f1))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and epoch_acc >= best_acc:
                best_f1 = epoch_f1
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        #print()

    time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,best_acc,best_f1

## Train the model
#lr=0.005701
#momentum=0.040802386
#batchSize=10
#decay=0.005001

def objective1(lr,momentum,decay):
    model_ft,criterion,  optimizer_ft,  exp_lr_scheduler, batchSize = load_model(lr,momentum,16,decay)  
   
    model_ft,best_acc,best_f1 = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,batchSize,
                       num_epochs=70)
    return model_ft,best_acc,best_f1
    


pbounds = {'lr': (0.05, 0.069), 'momentum': (0.001, 0.4), 'decay':(0.00001,0.01)}

bayesian_opt= False


if bayesian_opt:
    print('Hyperparameter tuning started: ')
    optimizer = BayesianOptimization(
            f=objective1,
            pbounds=pbounds,
            random_state=1,
     )

    optimizer.maximize(
           init_points=1,
           n_iter=20,
    )
    
#  0.625    |  16.0     |  0.0001   |  1e-06    |  0.99 
#  0.9306   |  8.728    |  0.003729 |  0.05634  |  0.06412
#  0.9167   |  11.95    |  0.0001   |  0.059    |  0.05
#  0.9167   |  5.678    |  0.0001   |  0.059    |  0.05




# linknet:       15.53    |  0.009325 |  0.005589 |  0.3748  
# objective1 (0.005589,0.3748,15,0.009325)

# DenseNet:       15.53    |  0.009325 |  0.005589 |  0.3748  
# objective1  (0.005589,0.3748,15,0.009325)

# previous mode: 15.62    |  0.007655 |  0.02955  |  0.4139
# objective1 (0.02955,0.4139,15,0.007655)

#  objective1 (0.006,0.05,0.0001)
model_ft,best_acc,best_f1 = objective1 (0.05,0.3748,0.009325 )
print('best_acc',best_acc)
print('best_f1',best_f1)
# Save the entire model
print("\nSaving the model...")
torch.save(model_ft, PATH)

'''
Sample run: python train.py --mode=finetue
'''
