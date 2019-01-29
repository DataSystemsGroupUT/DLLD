import pandas as pd
import numpy as np
import cv2
import os
base='/content/gdrive/My Drive/Data Systems/Segmentation/'
input_dir='Input/'
output_dir='B_box/'
info=pd.read_csv('/Users/tarunkhajuria/Dropbox/My files @Tarun Khajuria/University of Tartu/Big Data Group/Deep Lesion Project/Experiment/DL_info.csv')
lung_info=info[info.Coarse_lesion_type == 5] # 5 for lung lesions

for file in os.listdir(input_dir):
    if(file=='.DS_Store'):
        continue
    entry=lung_info[lung_info.File_name==file]
    try:
        img=cv2.imread(input_dir+file)
    except:
        print("cont")
        continue

    print(file)
    print(entry)
    b_box=np.array(entry.Bounding_boxes.values[0].split(',')).astype(np.float)
    b_box=b_box.astype(np.int)
    cv2.rectangle(img,(b_box[0],b_box[1]),(b_box[2],b_box[3]),(0,255,0),3)
    cv2.imwrite(output_dir+file,img)
    print(file)
