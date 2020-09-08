import os
import cv2
from glob import glob

root_folder = os. getcwd() +'/data/training'
import pickle

diction = {}

from random import shuffle

dir_images = []
dir_label = []

for subdir in os.listdir(root_folder):
        print('subdir',os.path.join(root_folder, subdir))
        for file_count, file_name in enumerate( sorted(glob(os.path.join(root_folder, subdir)+ '/*'),key=len) ):
            img = cv2.imread(file_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dir_images.append(img)
            dir_label.append(subdir)
            
        
dir_label_shuf = []
dir_images_shuf = []
index_shuf = list(range(len(dir_label)))
shuffle(index_shuf)
for i in index_shuf:
    dir_label_shuf.append(dir_label[i])
    dir_images_shuf.append(dir_images[i])    
        
diction['X_tr'] = dir_images_shuf
diction['y_tr'] = dir_label_shuf
        
       
print(len(diction['X_tr']))
print(diction['y_tr'])
print(diction['X_tr'][0].shape)

with open('training.pickle', 'wb') as handle:
    pickle.dump(diction, handle, protocol=pickle.HIGHEST_PROTOCOL)