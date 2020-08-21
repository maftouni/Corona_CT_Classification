import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

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
from sklearn.metrics import f1_score, accuracy_score
from torch import FloatTensor
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset 
from torch.nn import functional as F
import matplotlib.pyplot as plt


from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


from models import UNet11_Attention, LinkNet34_Attention, UNet, UNet16_Attention, AlbuNet, DenseNet169, DenseNet121_Attention,DenseNet121, DenseNet121_reduced, previous_model, DenseNet169_dropout




from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
import random


num_cpu = multiprocessing.cpu_count()
# Number of classes
num_classes = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Applying transforms to the data

image_transforms = { 
     'train': transforms.Compose([
        transforms.Resize((224, 224)),
    #transforms.CenterCrop(size=224),
    #transforms.Pad(25, padding_mode='symmetric'),     
    transforms.RandomHorizontalFlip(),
     transforms.RandomRotation(10),
    # random brightness and random contrast
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
     ]),
     'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
     ])
    }

# Set the train and validation directory paths
train_directory = 'data/training'
test_directory = 'data/test'

dataset = datasets.ImageFolder(root=train_directory,transform=image_transforms['train'])

dataset_test = datasets.ImageFolder(root=test_directory,transform=image_transforms['valid'])
       

# Fitting the classifier
def fit_classifier():
 
        torch.manual_seed(8)
        torch.cuda.manual_seed(8)
        np.random.seed(8)
        random.seed(8)
   
       #clf = svm.SVC()
       #clf = RandomForestClassifier(n_estimators=100)
       
       #clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0) 
       #clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
       #clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    
       #clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
       #clf = svm.SVC(kernel='rbf', probability=True)
        clf = KNeighborsClassifier(n_neighbors=50)
       #clf = AdaBoostClassifier(n_estimators=100)
       #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

  

        model = DenseNet121(num_classes,pretrained=True)
    #DenseNet169_dropout(models.densenet169(pretrained=True),num_classes)
        checkpoint0 = torch.load('model_DenseNet_pretrained_final.pth', map_location='cpu')
        model.load_state_dict(checkpoint0.module.state_dict()) 
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Identity() 
       
        for param in model.parameters():
             param.requires_grad_(False) 
       
       
        model = nn.DataParallel(model, device_ids=[0,1,2]).cuda() 
    
       
        dataset_sizes = {
    'train':len(dataset),
    'test':len(dataset_test)
}
        print('len(datasets[train])', len(dataset))
        print('len(dataset_test)', len(dataset_test))
        
        
        dataloaders = {
        'train' : data.DataLoader(dataset, batch_size=len(dataset), shuffle=True,
                            num_workers=num_cpu, pin_memory=True, worker_init_fn=np.random.seed(8), drop_last=False),
        'test' : data.DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=True,
                            num_workers=num_cpu, pin_memory=True, worker_init_fn=np.random.seed(8), drop_last=False)   
}


        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 0
        best_loss = 100000
    
        # Tensorboard summary
        writer = SummaryWriter()
    
        for epoch in range(1):
            #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            #print('-' * 10)

        # Each epoch has a training and validation phase
            jj=0
            all_best_accs = {}
            all_best_f1s = {}
            for phase in ['train','test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                  
                    outputs = model(inputs)
                 # fit the classifier on training set and then predict on test 
                    if phase == 'train': 
                         clf.fit(outputs.cpu(), labels.cpu())
                         all_best_accs[phase]=accuracy_score(labels.cpu(), clf.predict(outputs.cpu()))
                         all_best_f1s[phase]= f1_score(labels.cpu(), clf.predict(outputs.cpu()))
                         print(phase, ' ',accuracy_score(labels.cpu(), clf.predict(outputs.cpu())))   
                    if phase != 'train' :
                         predict = clf.predict(outputs.cpu())
                         all_best_accs[phase]=accuracy_score(labels.cpu(), clf.predict(outputs.cpu()))
                         all_best_f1s[phase]= f1_score(labels.cpu(), clf.predict(outputs.cpu()))
                         print(phase, ' ',accuracy_score(labels.cpu(), clf.predict(outputs.cpu())))    
 
        return all_best_accs,all_best_f1s


    
best_acc,best_f1 = fit_classifier()

print('best_acc',best_acc)
print('best_f1',best_f1)

