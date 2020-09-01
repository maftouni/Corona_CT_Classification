import os
import cv2
from glob import glob
#print(os. getcwd() )
root_folder = os. getcwd() +'/data/test'
print(root_folder)
import pickle

diction = {}

from random import shuffle

dir_images = []
dir_label = []
#for root, dirs, files in os.walk(root_folder):
for subdir in os.listdir(root_folder):
        print('subdir',os.path.join(root_folder, subdir))
        #all_images=glob.glob(os.path.join(root, subdir)+ '/*')
        for file_count, file_name in enumerate( sorted(glob(os.path.join(root_folder, subdir)+ '/*'),key=len) ):
            #print (file_name)
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

with open('test.pickle', 'wb') as handle:
    pickle.dump(diction, handle, protocol=pickle.HIGHEST_PROTOCOL)


#print(len(images['2corona']),len(images['1noncorona']))
               
                
    

#print('hellohello')    
# Print out the content dict    
#for folder, filenames in content.items():
    #print ('Folder: {}'.format(folder))
    #print ('Filenames:')
    #for filename in filenames:
    #     print ('-> {}'.format(filename))