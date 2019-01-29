base_input='Final Test Org/'
base_output='Output15X15/'
models='Model/'


import imageio as io
import tensorflow as tf
import numpy as np
import pickle
import time
import os
from tfrbm import GBRBM
from sklearn.ensemble import RandomForestClassifier

patch_size=15
f=open(models+'model_15.pickle','rb')
model=pickle.load(f)
rbm=GBRBM(patch_size*patch_size, 60, learning_rate=0.001, momentum=0, err_function='mse', use_tqdm=True, sample_visible=False, sigma=1)
rbm.set_weights(model['rbm'][0],model['rbm'][1],model['rbm'][2])
rf=model['RF']

def predict(data):
    features=rbm.transform(data)
    return rf.predict(features)

def image_segment(img,patch_size):
    mat=np.array(img)
    img=np.array(img)
    img=(img-img.mean())/img.std()
    mid=int(patch_size/2)
    for i in range(mid,img.shape[0]-mid-1):
        row=[]
        for j in range(mid,img.shape[1]-mid-1):
            patch=img[i-mid:i+mid+1,j-mid:j+mid+1].flatten()
            row.append(patch)
        row=np.array(row)
        result=predict(row)
        for ind,out in enumerate(result):    
            if(out=='L'):
                mat[i,ind+mid]=0
            else:
                mat[i,ind+mid]=255
    return mat

files=os.listdir(base_output)
for ind,file in enumerate(os.listdir(base_input)):
    if(file in files):
        continue
    try:
        image=io.imread(base_input+file)
    except:
        continue
    print(ind)
    segmented=image_segment(image,patch_size)
    io.imwrite(base_output+file,segmented)



