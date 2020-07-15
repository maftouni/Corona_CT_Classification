import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os 
from torch import FloatTensor

# Paths for image directory and model
IMDIR='data/test/1noncorona/'
#sys.argv[1]
MODEL='model.pth'
#'models/resnet18.pth'

# Load the model for testing
model = torch.load(MODEL)
model.eval()

# Class labels for prediction
class_names=['noncorona','corona']

# Retreive 9 random images from directory
#files=Path(IMDIR).resolve().glob('*.*')

#images=random.sample(list(files), 3)


# list all files in dir
print(os.listdir(IMDIR),len(os.listdir(IMDIR)))
files = [f for f in os.listdir(IMDIR)]
print('filesfilesfilesfiles',files,len(files))


# select 0.1 of the files randomly 
images = files
#np.random.choice(files, int(len(files)*1))
print('int(len(files)*1',int(len(files)*1))
# Configure plots
fig = plt.figure(figsize=(40,int(len(files)*0.15)))
rows,cols = int(len(files)*1)/13,13

# Preprocessing transformations
preprocess=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cpu")
#"cuda:0" if torch.cuda.is_available() else 

predictions=FloatTensor()
# Perform prediction and plot results
with torch.no_grad():
    for num,img in enumerate(images):
         img=Image.open(IMDIR+img).convert('RGB')
         inputs=preprocess(img).unsqueeze(0).to(device)
         outputs = model(inputs)
         _, preds = torch.max(outputs, 1) 
         predictions=torch.cat([predictions,preds.float()])
         label=class_names[preds]
         plt.subplot(rows,cols,num+1)
         plt.title("Pred: "+label)
         plt.axis('off')
         plt.imshow(img)
    plt.savefig('books_read.png')

print(predictions, predictions.sum())

'''
Sample run: python test.py test
'''
