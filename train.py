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
from PIL import Image

#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, help="Training mode: finetue/transfer/scratch")
args= vars(ap.parse_args())

# Set training mode
train_mode=args["mode"]

# Set the train and validation directory paths
train_directory = '/u/ml00_s/zhou950/DenseNet_Pytorch/Data/TrainingImages'
valid_directory = '/u/ml00_s/zhou950/DenseNet_Pytorch/Data/ValidImages'
test_directory = '/u/ml00_s/zhou950/DenseNet_Pytorch/Data/TestImages'
# Set the model save path
PATH = "model.pth"

# Batch size
bs = 10
# Number of epochs
num_epochs = 20
# Number of classes
num_classes = 2
# Number of workers
num_cpu = multiprocessing.cpu_count()



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
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(data.Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample

# Load data from folders
#dataset = {
#    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
#    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
#    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
#}

dataset = {
    'train': CovidCTDataset(root_dir='/u/ml00_s/zhou950/DenseNet_Pytorch/Images-processed',
                            txt_COVID='/u/ml00_s/zhou950/DenseNet_Pytorch/Data-split/COVID/trainCT_COVID.txt',
                            txt_NonCOVID='/u/ml00_s/zhou950/DenseNet_Pytorch/Data-split/NonCOVID/trainCT_NonCOVID.txt',
                            transform=image_transforms['train']),
    'valid': CovidCTDataset(root_dir='/u/ml00_s/zhou950/DenseNet_Pytorch/Images-processed',
                            txt_COVID='/u/ml00_s/zhou950/DenseNet_Pytorch/Data-split/COVID/valCT_COVID.txt',
                            txt_NonCOVID='/u/ml00_s/zhou950/DenseNet_Pytorch/Data-split/NonCOVID/valCT_NonCOVID.txt',
                            transform=image_transforms['valid']),
    'test': CovidCTDataset(root_dir='/u/ml00_s/zhou950/DenseNet_Pytorch/Images-processed',
                           txt_COVID='/u/ml00_s/zhou950/DenseNet_Pytorch/Data-split/COVID/testCT_COVID.txt',
                           txt_NonCOVID='/u/ml00_s/zhou950/DenseNet_Pytorch/Data-split/NonCOVID/testCT_NonCOVID.txt',
                           transform=image_transforms['test'])
}

# Size of train and validation data
dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid']),
    'test':len(dataset['test'])
}

# Create iterators for data loading
dataloaders = {
    'train':data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=False),
    'valid':data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=False),
    'test':data.DataLoader(dataset['test'], batch_size=bs, shuffle=True,
                            num_workers=num_cpu, pin_memory=True, drop_last=False)
}

# Class names or target labels
class_names = dataset['train'].classes
print("Classes:", class_names)
 
# Print the train and validation data sizes
print("Training-set size:",dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])

# Set default device as gpu, if available
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if train_mode=='finetune':
    # Load a pretrained model - Resnet18
    print("\nLoading resnet18 for finetuning ...\n")
    model_ft = models.resnet50(pretrained=True)

    # Modify fc layers to match num_classes
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes )

elif train_mode=='densenet':
    model_ft = models.densenet169(pretrained=True)
    
    # Modify fc layers to match num_classes
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs,num_classes )
    
elif train_mode=='scratch':
    # Load a custom model - VGG11
    print("\nLoading VGG11 for training from scratch ...\n")
    model_ft = MyVGG11(in_ch=3,num_classes=11)

    # Set number of epochs to a higher value
    num_epochs = 100

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
model_ft = model_ft.to(device)

pretrained_net = torch.load('/u/ml00_s/zhou950/COVID-CT/baseline methods/Self-Trans/Self-Trans.pt')


model_ft.load_state_dict(pretrained_net)


# Print model summary
print('Model Summary:-\n')
for num, (name, param) in enumerate(model_ft.named_parameters()):
    print(num, name, param.requires_grad )
#summary(model_ft, input_size=(3, 224, 224))
print(model_ft)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer 
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

# Learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=10)


# Model training routine 
print("\nTraining:-\n")
def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            predictions=FloatTensor().to(device)
            all_labels=FloatTensor().to(device)
            #torch.tensor([0.])

            # Iterate over data.
            for inputs, label in enumerate(dataloaders[phase]):
                #inputs = inputs.to(device, non_blocking=True)
                #labels = labels.to(device, non_blocking=True)
                inputs = label['img'].to(device)#, non_blocking=True)
                labels = label['label'].to(device)#, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    #print('preds.float()',preds.float())
                    #print('predictions',predictions)
                    predictions = torch.cat([predictions,preds.float()])
                    all_labels = torch.cat([all_labels,labels.float()])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                
                
            if phase == 'train':
                scheduler.step()
                

            print('all_labels',all_labels.tolist())
            print('predictions',predictions.tolist())
            epoch_f1=f1_score(all_labels.tolist(), predictions.tolist())
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f} f1: {:.4f}'.format(
                phase, epoch_loss, epoch_acc,epoch_f1))

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
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    phase = 'test'

    model.eval()  # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    predictions = FloatTensor().to(device)
    all_labels = FloatTensor().to(device)
    # torch.tensor([0.])

    # Iterate over data.
    for inputs, label in enumerate(dataloaders['test']):
        inputs = label['img'].to(device, non_blocking=True)
        labels = label['label'].to(device, non_blocking=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # print('preds.float()',preds.float())
            # print('predictions',predictions)
            predictions = torch.cat([predictions, preds.float()])
            all_labels = torch.cat([all_labels, labels.float()])


        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    print('all_labels', all_labels.tolist())
    print('predictions', predictions.tolist())
    epoch_f1 = f1_score(all_labels.tolist(), predictions.tolist())
    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]

    print('{} Loss: {:.4f} Acc: {:.4f} f1: {:.4f}'.format(
            phase, epoch_loss, epoch_acc, epoch_f1))

    return model

# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
# Save the entire model
print("\nSaving the model...")
torch.save(model_ft, PATH)

'''
Sample run: python train.py --mode=finetue
'''
