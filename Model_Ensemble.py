import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


import pickle
from datetime import datetime
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import PIL
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time, copy, argparse
import multiprocessing
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from torch import FloatTensor
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset 
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader



from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import joblib

from torch import nn
import torch
from torchvision import models
import functools
from torch.autograd import Variable
import torchvision
from torch.nn import functional as F
from torch.nn import init
from basic_layers import *
from attention_module import *







def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 6, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.batch = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)
        
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.conv(x)
        #print(list(x.type()))
        x= self.batch(x)
        x = self.activation(x)
        return x
    
class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )
            
            for m in self.children():
                 init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.block(x)


class UnetUp2_CT(nn.Module):
    def __init__(self,in_size1,in_size2, out_size, is_batchnorm=True):
        super(UnetUp2_CT, self).__init__()
        self.conv = conv_block(in_size1+in_size2, out_size)
        self.up = nn.Upsample(scale_factor=2)
        
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        out = self.conv(torch.cat([inputs1, outputs2], 1))
        return out
   
    
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):

        x = self.conv(x)
        return x    
    
    
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        x = self.up(x)
        return x    
    

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class ResidualAttentionModel(nn.Module):
    # for input size 32
    def __init__(self,n_classes):
        
        super(ResidualAttentionModel, self).__init__()
        input_size = 224
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32
        # self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 16*16
        self.residual_block1 = ResidualBlock(32, 128)  # 32*32
        self.attention_module1 = AttentionModule_stage1_cifar(128, 128, size1=(input_size, input_size), size2=(int(input_size/2), int(input_size/2)))  # 32*32
        self.residual_block2 = ResidualBlock(128, 256, 2)  # 16*16
        self.attention_module2 = AttentionModule_stage2_cifar(256, 256, size=(int(input_size/2), int(input_size/2)))  # 16*16
        self.attention_module2_2 = AttentionModule_stage2_cifar(256, 256, size=(int(input_size/2), int(input_size/2)))  # 16*16 # tbq add
        self.residual_block3 = ResidualBlock(256, 512, 2)  # 4*4
        self.attention_module3 = AttentionModule_stage3_cifar(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.attention_module3_3 = AttentionModule_stage3_cifar(512, 512)  # 8*8 # tbq add
        self.residual_block4 = ResidualBlock(512, 1024)  # 8*8
        self.residual_block5 = ResidualBlock(1024, 1024)  # 8*8
        self.residual_block6 = ResidualBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=int(input_size/4))
        )
        self.fc = nn.Linear(1024,n_classes)

    def forward(self, x):
        #print('xxxxx',list(x.shape))
        out = self.conv1(x)
        # out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        #print('outttttttttttttttt',list(out.shape))
        out = self.attention_module1(out)
        #print('outttttttttttt_atte',list(out.shape))
        out = self.residual_block2(out)
        out = self.attention_module2(out)
        out = self.attention_module2_2(out)
        out = self.residual_block3(out)
        # print(out.data)
        out = self.attention_module3(out)
        out = self.attention_module3_2(out)
        out = self.attention_module3_3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    
    
class DenseNet121(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,is_batchnorm=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        self.is_batchnorm=is_batchnorm
        filters = [64, 128, 256, 512,1024]
        densenet = models.densenet121(pretrained=pretrained)

        self.firstconv = densenet.features.conv0
        self.firstbn = densenet.features.norm0
        self.firstrelu = densenet.features.relu0
        self.firstmaxpool = densenet.features.pool0
        self.encoder1 = densenet.features.denseblock1
        self.transition1=densenet.features.transition1
        self.encoder2 = densenet.features.denseblock2
        self.transition2=densenet.features.transition2
        self.encoder3 = densenet.features.denseblock3
        self.transition3=densenet.features.transition3
        self.encoder4 = densenet.features.denseblock4
        self.norm5=densenet.features.norm5
        
        self.num_ftrs = densenet.classifier.in_features
        self.classifier = nn.Linear(self.num_ftrs,2 )
        
        
       
    
        
    
    def forward(self, x):
        
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e2 = self.encoder1(e1)
        e3 = self.transition1(e2)
        e3 = self.encoder2(e3)
        e4 = self.transition2(e3)
        e4 = self.encoder3(e4)
        e5 = self.transition3(e4)
        e5 = self.encoder4(e5)
        e5 = self.norm5(e5)
        e5= F.adaptive_avg_pool2d(e5, (1, 1)).view(e5.shape[0], -1)
        #print('ee5',list(e5.shape))
    
        x_out = self.classifier(e5)
        #print("shape of x_out",list(x_out .shape))
         
        return x_out    

    
class DenseNet169(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True,is_batchnorm=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        self.is_batchnorm=is_batchnorm
        densenet = models.densenet169(pretrained=pretrained)

        self.firstconv = densenet.features.conv0
        self.firstbn = densenet.features.norm0
        self.firstrelu = densenet.features.relu0
        self.firstmaxpool = densenet.features.pool0
        self.encoder1 = densenet.features.denseblock1
        self.transition1=densenet.features.transition1
        self.encoder2 = densenet.features.denseblock2
        self.transition2=densenet.features.transition2
        self.encoder3 = densenet.features.denseblock3
        self.transition3=densenet.features.transition3
        self.encoder4 = densenet.features.denseblock4
        self.norm5=densenet.features.norm5
        
        self.num_ftrs = densenet.classifier.in_features
        self.classifier = nn.Linear(self.num_ftrs,num_classes )
     
    
    def forward(self, x):
        
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e1 = self.firstmaxpool(x)
        e2 = self.encoder1(e1)
        e3 = self.transition1(e2)
        e3 = self.encoder2(e3)
        e4 = self.transition2(e3)
        e4 = self.encoder3(e4)
        e5 = self.transition3(e4)
        e5 = self.encoder4(e5)
        e5 = self.norm5(e5)
        e5= F.adaptive_avg_pool2d(e5, (1, 1)).view(e5.shape[0], -1)
        #print('ee5',list(e5.shape))
    
        x_out = self.classifier(e5)
        #print("shape of x_out",list(x_out .shape))
         
        return x_out
    


classes = ('1noncorona', '2corona')

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(55, 55))
    for idx in np.arange(16):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        
        return x, y
    
    def __len__(self):
        return len(self.data)

    
class MyDataset_test(Dataset):
    def __init__(self, data, transform=None):
        self.data = torch.from_numpy(data).float()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        
        if self.transform:
            x = self.transform(x)
            
        
        return x
    
    def __len__(self):
        return len(self.data)


def linear_to_identity(network):
    for layer in network.children():
        if type(layer) == nn.Linear: # if sequential layer, apply recursively to layers in sequential layer
            layer = nn.Identity()
    return network        
        
            
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB,a,b, nb_classes=2):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        #self.modelC = modelC
        #self.modelD = modelD
        
        #self.num_ftrsA = modelA.classifier.in_features
        #self.num_ftrsB = modelB.fc.shape
        
        #num_ftrsC = modelC.fc.in_features
        #num_ftrsD = modelD.classifier.in_features
        # Remove last linear layer
        self.modelA.classifier = nn.Identity()
        self.modelB.fc = nn.Identity()
        #self.modelC.fc = nn.Identity()
        #self.modelD.classifier = nn.Identity()
        
        # Create new classifier
        
        self.classifier = nn.Linear(a+b, 755)
        self.classifier2 = nn.Linear(755, nb_classes)
        #self.classifier3 = nn.Linear(30, nb_classes)
        
    def forward(self, x):
        
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        #print(x2.shape)
        x2 = x2.view(x2.size(0), -1)
        
        #x3 = self.modelC(x)
        #x3 = x3.view(x3.size(0), -1)
        
        #x4 = self.modelD(x)
        #x4 = x4.view(x4.size(0), -1)
        
        x = torch.cat((x1, x2), dim=1)
        #print('x1.shape',x1.shape)
        #print('x2.shape',x2.shape)
        #print('x.shape',x.shape)
        #print(self.num_ftrsA+self.num_ftrsA)
        x = self.classifier(F.relu(x))
        x = self.classifier2(F.relu(x))
        #x = self.classifier3(F.relu(x))
        return x    
    
    

def estimate(X_train,y_train,X_test,y_test):
    i = 0
    ii = 0
    nrows=256
    ncolumns=256
    channels=1
    ntrain=0.85*len(X_train)
    nval=0.15*len(X_train)
    batch_size=20
    epochs=15
    # Number of classes
    num_cpu = multiprocessing.cpu_count()
    num_classes = 2
    torch.manual_seed(8)
    torch.cuda.manual_seed(8)
    np.random.seed(8)
    random.seed(8)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    
    X_t = []
    X_test=np.reshape(np.array(X_test),[len(X_test),])
    
    for img in list(range(0,len(X_test))):
        if X_test[img].ndim>=3:
            X_t.append(np.moveaxis(cv2.resize(X_test[img][:,:,:3], (nrows,ncolumns), interpolation=cv2.INTER_CUBIC), -1, 0))
        else:
            smimg= cv2.cvtColor(X_test[img],cv2.COLOR_GRAY2RGB)
            X_t.append(np.moveaxis(cv2.resize(smimg, (nrows,ncolumns), interpolation=cv2.INTER_CUBIC), -1, 0))
        
        if y_test[img]=='COVID':
            y_test[img]=1
        elif y_test[img]=='NonCOVID' :
            y_test[img]=0
        else:
            continue

    x_test = np.array(X_t)
    y_test = np.array(y_test)
    
    
    X = []
    X_train=np.reshape(np.array(X_train),[len(X_train),])
    for img in list(range(0,len(X_train))):
        if X_train[img].ndim>=3:
            X.append(np.moveaxis(cv2.resize(X_train[img][:,:,:3], (nrows,ncolumns),interpolation=cv2.INTER_CUBIC), -1, 0))
        else:
            smimg= cv2.cvtColor(X_train[img],cv2.COLOR_GRAY2RGB)
            X.append(np.moveaxis(cv2.resize(smimg, (nrows,ncolumns),interpolation=cv2.INTER_CUBIC), -1, 0))
        
        if y_train[img]=='COVID':
            y_train[img]=1
        elif y_train[img]=='NonCOVID' :
            y_train[img]=0
        else:
            continue

    x = np.array(X)
    y_train = np.array(y_train)
    
    
    outputs_all = []
    labels_all = []
    
    X_train, X_val, y_train, y_val = train_test_split(x, y_train, test_size=0.15, random_state=0)
    
    
    
    image_transforms = { 
     'train': transforms.Compose([
         transforms.Lambda(lambda x: x/255),
        transforms.ToPILImage(), 
         #transforms.Resize((256, 256)),
         transforms.Resize((230, 230)),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),     
    #transforms.CenterCrop(size=224),
    #transforms.Pad(25,fill=0, padding_mode='constant'),     
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    #transforms.Affine(10,shear =(0.1,0.1)),
    # random brightness and random contrast
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.45271412, 0.45271412, 0.45271412],
                             [0.33165374, 0.33165374, 0.33165374])
     ]),
     'valid': transforms.Compose([
         transforms.Lambda(lambda x: x/255),
         transforms.ToPILImage(), 
       transforms.Resize((230, 230)),
    #     transforms.Resize((230, 230)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.45271412, 0.45271412, 0.45271412],
                             [0.33165374, 0.33165374, 0.33165374])
     ])
    }
    
    
    
    train_data = MyDataset(X_train, y_train,image_transforms['train'])
    
    valid_data = MyDataset(X_val, y_val,image_transforms['valid'])
    
    test_data = MyDataset(x_test, y_test,image_transforms['valid'])
    
    dataset_sizes = {
    'train':len(train_data),
    'valid':len(valid_data),
    'test':len(test_data)
}
    
    dataloaders = {
        'train' : data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, worker_init_fn=np.random.seed(7), drop_last=False),
        'valid' : data.DataLoader(valid_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, worker_init_fn=np.random.seed(7), drop_last=False),
        'test' : data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, worker_init_fn=np.random.seed(7), drop_last=False)   
}
    
    #model = ResidualAttentionModel(10)
    #checkpoint0 = torch.load('../model_resAttention.pth')
    #    
    #model.load_state_dict(checkpoint0)
    #num_ftrs = model.fc.in_features
    #    #model_ft.classifier = nn.Linear(num_ftrs,num_classes )
    #model.fc = nn.Linear(num_ftrs,2 )
    #    #model_ft = nn.Sequential(model_ft, nn.Linear(755,30 ),nn.Linear(30,num_classes ))
    
    ##model = DenseNet169(num_classes,pretrained=True)
     
    modelA = DenseNet121(num_classes,pretrained=True)
    num_ftrs1 = modelA.classifier.in_features
       #print('num_ftrsnum_ftrsnum_ftrs',num_ftrs1) 
       #modelA.classifier = nn.Linear(num_ftrs,num_classes)
        
    checkpoint0 = torch.load('Model_121_state.pth', map_location='cpu')
    modelA.load_state_dict(checkpoint0) 
        
    modelC = ResidualAttentionModel(2)
    #modelC = DenseNet169(num_classes,pretrained=True)
       #modelA = models.densenet169(pretrained=True)
    num_ftrs2 = modelC.fc.in_features
       #print('num_ftrsnum_ftrsnum_ftrs',num_ftrs1) 
       #modelA.classifier = nn.Linear(num_ftrs,num_classes)
        
    checkpoint0 = torch.load('Model_residual_state.pth', map_location='cpu')
       
    modelC.load_state_dict(checkpoint0)
    
    
       
       
       
    model = MyEnsemble(modelA, modelC,num_ftrs1,num_ftrs2)
    
    
    for param in modelC.parameters():
            param.requires_grad_(False) 
        
    for param in modelA.parameters():
             param.requires_grad_(False) 
    
   # checkpoint0 = torch.load('../modelss/model_fold1_state.pth', map_location='cpu')
    
   # model.load_state_dict(checkpoint0) 
    
    model = nn.DataParallel(model, device_ids=[ 0, 1,2, 3]).cuda()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.006775, momentum=0.5518,weight_decay=0.000578)
    #optimizer = optim.SGD(model.parameters(), lr=0.006775, momentum=0.5518,weight_decay=0.000578)
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.05)
    #scheduler =  lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)
    
   
    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = 0
    best_loss = 100000
    since = time.time()
    writer = SummaryWriter()
    
    model.train()
    
    for epoch in range(epochs): 
            print('epoch',epoch)
            jj=0
            for phase in ['train', 'valid','test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                predictions=FloatTensor()
                all_labels=FloatTensor()
                #torch.tensor([0.])
                # dataloaders,dataset_sizes = d        ata_loader(train_directory,valid_directory)

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    #inputs = inputs.to(device, non_blocking=True)
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
                    
                    #print('loss',loss)
                    #print('predictions',predictions)
                         predictions=torch.cat([predictions,preds.float()])
                         all_labels=torch.cat([all_labels,labels.float()])
                    
                         #a = list(model.parameters())[0].clone()

                    # backward + optimize only if in training phase
                         if phase == 'train':
                        #a0 = list(model.parameters())[0].clone()
                               loss.backward()
                               optimizer.step()
                        
                               #b = list(model.parameters())[0].clone()
                               #print('check training',torch.equal(a.data, b.data))
                                
                         if phase == 'train':
                                     jj+= 1
                                    
                                     if len(inputs) >=16 :
                                            
                                             #print('len(inputs)',len(inputs),i)
                                             writer.add_figure('predictions vs. actuals epoch '+str(epoch)+' '+str(jj) ,
                                             plot_classes_preds(model, inputs, labels))
                                            
                           
                        

                # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                
                
                if phase == 'train':
                    scheduler.step()
                

                #print('all_labels',all_labels.tolist())
                #print('predictions',predictions.tolist())
                epoch_f1=f1_score(all_labels.tolist(), predictions.tolist(),average='weighted')
                #print('epoch_f1',epoch_f1)
                print(phase, 'confusion_matrix',confusion_matrix(all_labels.tolist(), predictions.tolist()))
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = accuracy_score(all_labels.tolist(), predictions.tolist())
                #running_corrects.double() / dataset_sizes[phase]
                

                #print('{} Loss: {:.4f} Acc: {:.4f} f1: {:.4f}'.format(
                #phase, epoch_loss, epoch_acc,epoch_f1))

                # Record training loss and accuracy for each phase
                if phase == 'train':
                    writer.add_scalar('Train/Loss', epoch_loss, epoch)
                    writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                
                    writer.flush()
                elif phase == 'valid':
                    writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                    writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                    writer.flush()
                elif phase == 'test':
                    i+= 1
                    writer.add_scalar('Test/Loss', epoch_loss, epoch)
                    writer.add_scalar('Test/Accuracy', epoch_acc, epoch)
                    writer.flush()           

            # deep copy the model
                if phase == 'valid' and epoch_loss <= best_loss:
                
                    best_f1 = epoch_f1
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.module.state_dict())
                    best_model_wts_module = copy.deepcopy(model.state_dict())
                #model.load_state_dict(best_model_wts)
                #torch.save(model, "modelss/model"+str(epoch)+".pth" )

        #print()
    #model = model.cpu() 
    model.load_state_dict(best_model_wts_module)
    torch.save(model, "Model.pth")
    torch.save(best_model_wts,"Model_state.pth")
    
        #model_init.save_state_dict("modelss/model_fold"+str(fold)+".pth" )
        #torch.save({
        #    'epoch': best_epoch,
        #    'model_state_dict': best_model_wts,
        #    'optimizer_state_dict': optimizer.state_dict(),
        #    'loss': best_loss}, "modelss/model_fold"+str(fold)+".pth" )

    time_elapsed = time.time() - since
        
        #secondary = list(model.parameters())[0].clone()
        #print('check training main2',torch.equal(intial.data, secondary.data))
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    print('Best valid f1: {:4f}'.format(best_f1))
    print('best epoch: ', best_epoch)
        
    model.module.classifier2 = nn.Identity()
    #model.layer[-1] = 
    #print(model)
    for param in model.parameters():
             param.requires_grad_(False)
            
    clf1 = svm.SVC(kernel='rbf', probability=True)
    all_best_accs = {}
    all_best_f1s = {}
    clf2 = ExtraTreesClassifier(n_estimators=40, max_depth=None, min_samples_split=30, random_state=0)
    
    for phase in ['train','test']:
                outputs_all = []
                labels_all = []
                model.eval()   # Set model to evaluate mode

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                  
                    outputs = model(inputs)
                    #print(outputs.shape)
                    outputs_all.append(outputs)
                    labels_all.append(labels)
                    
                outputs = torch.cat(outputs_all)
                #print('outputss',outputs.shape)
                labels = torch.cat(labels_all)
                
                 # fit the classifier on training set and then predict on test 
                if phase == 'train': 
                         clf1.fit(outputs.cpu(), labels.cpu())
                         clf2.fit(outputs.cpu(), labels.cpu())
                         filename1 = 'classifier_SVM.sav'
                         filename2 = 'classifier_ExtraTrees.sav'
                        
                         joblib.dump(clf1, filename1)
                         joblib.dump(clf2, filename2)
                         all_best_accs[phase]=accuracy_score(labels.cpu(), clf1.predict(outputs.cpu()))
                         all_best_f1s[phase]= f1_score(labels.cpu(), clf1.predict(outputs.cpu()))
                         #print(phase, ' ',accuracy_score(labels.cpu(), clf.predict(outputs.cpu())))   
                         print(phase, 'confusion_matrix of SVM',confusion_matrix(labels.cpu(), clf1.predict(outputs.cpu())))   
                         print(phase, 'confusion_matrix of ExtraTrees',confusion_matrix(labels.cpu(), clf2.predict(outputs.cpu())))   
                if phase == 'test' :
                         predict = clf1.predict(outputs.cpu())
                         all_best_accs[phase]=accuracy_score(labels.cpu(), clf1.predict(outputs.cpu()))
                         all_best_f1s[phase]= f1_score(labels.cpu(), clf.predict(outputs.cpu()))
                         print(phase, 'confusion_matrix of SVM',confusion_matrix(labels.cpu(), clf1.predict(outputs.cpu())))   
                         print(phase, 'confusion_matrix of ExtraTrees',confusion_matrix(labels.cpu(), clf2.predict(outputs.cpu()))) 
                         #print(phase, ' ',accuracy_score(labels.cpu(), clf.predict(outputs.cpu())))  
                            
    print('Best Acc: ',all_best_accs)
    print('Best f1: ',all_best_f1s)
    
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model
        
        
        
        
    

def predict(X_test,model_main=None):
  
    i = 0
    nrows=256
    ncolumns=256
    num_classes = 2
    bs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelA = DenseNet121(num_classes,pretrained=True)
    num_ftrs1 = modelA.classifier.in_features
        
    modelB = ResidualAttentionModel(2)
    #DenseNet169(num_classes,pretrained=True)
    
    num_ftrs2 = modelB.fc.in_features
      
    model_main = MyEnsemble(modelA, modelB,num_ftrs1,num_ftrs2)
    checkpoint0 = torch.load("Model_state.pth")
    model_main.load_state_dict(checkpoint0)
    
    for param in model_main.parameters():
             param.requires_grad_(False) 
     
    model_main = nn.DataParallel(model_main, device_ids=[ 0, 1,2, 3]).cuda()
    X_t = []
    X_test=np.reshape(np.array(X_test),[len(X_test),])
    
    for img in list(range(0,len(X_test))):
        if X_test[img].ndim>=3:
            X_t.append(np.moveaxis(cv2.resize(X_test[img][:,:,:3], (nrows,ncolumns), interpolation=cv2.INTER_CUBIC), -1, 0))
        else:
            smimg= cv2.cvtColor(X_test[img],cv2.COLOR_GRAY2RGB)
            X_t.append(np.moveaxis(cv2.resize(smimg, (nrows,ncolumns), interpolation=cv2.INTER_CUBIC), -1, 0))
       

    x = np.array(X_t)
    y_pred=[]
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cpu")
    
    #model.classifier2 = nn.Linear(755, 2)
    #model.load_state_dict(torch.load('test_state_dict_new.pth'))
    #model.classifier2 = nn.Identity()    
    
   
    model_main.eval()
    
    image_transforms = transforms.Compose([
        transforms.Lambda(lambda x: x/255),
        transforms.ToPILImage(), 
            transforms.Resize((230, 230)),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.45271412, 0.45271412, 0.45271412],
                             [0.33165374, 0.33165374, 0.33165374])
     ])
    
    
    dataset = MyDataset_test(x,image_transforms)
    dataloader = DataLoader(
    dataset,
    batch_size=bs,
    pin_memory=True,worker_init_fn=np.random.seed(0), drop_last=False)
    
    
    for inputs in dataloader:
        #inputs = torch.from_numpy(inputs).float()
        inputs = inputs.to(device, non_blocking=True)
        outputs = model_main(inputs)
        _, preds = torch.max(outputs, 1)
        
        #pred = clf.predict(outputs.cpu())
        for ii in range(len(preds)):
           if preds[ii] > 0.5:
               y_pred.append('COVID')
           
           else:
               y_pred.append('NonCOVID')
            
        i+=1
        if i% math.ceil(len(X_test)/bs)==0:
               break
    
    model_main.module.classifier2 = nn.Identity()
    
    
    clf = loaded_model = joblib.load('classifier_ExtraTrees.sav')
    for param in model_main.parameters():
             param.requires_grad_(False)
    y_pred2=[]
    for inputs in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model_main(inputs)
        preds = clf.predict(outputs.cpu())
        
        for ii in range(len(preds)):
           if preds[ii] > 0.5:
               y_pred2.append('COVID')
            
           else:
               y_pred2.append('NonCOVID')
           
        i+=1
        if i% math.ceil(len(X_test)/bs)==0:
               break
                
    return y_pred,y_pred2


#dbfile = open('training.pickle', 'rb')      
#db = pickle.load(dbfile) 
#print(db)
#print(len(np.array(db['y_tr'])))
#print(np.array(db['X_tr'])[0].shape)

#dbfile = open('test.pickle', 'rb')      
#db_test = pickle.load(dbfile) 

#model = estimate(db['X_tr'],db['y_tr'],db_test['X_tr'],db_test['y_tr'])

#model = torch.load("hopefully.pth")
#dbfile = open('training.pickle', 'rb')      
#db_test = pickle.load(dbfile) 

            
            
#y_pred,y_pred2 = predict(db_test['X_tr'])
#print(y_pred)
#print(db_test['y_tr'])
#acc= accuracy_score(db_test['y_tr'], y_pred)
#acc2= accuracy_score(db_test['y_tr'], y_pred2)
#print(confusion_matrix(db_test['y_tr'], y_pred))
#print(confusion_matrix(db_test['y_tr'], y_pred2))

#print(acc,acc2)
