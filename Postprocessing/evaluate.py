
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.data import data_dir
from skimage.util import img_as_ubyte
from skimage import io
from skimage.morphology import disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
import numpy as np
from skimage import segmentation
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage import morphology
import pickle
from skimage import measure

from skimage import segmentation
from skimage import filters
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import pandas as pd
import imageio as io

def eval_box(bbox1,bbox2):
    bbox1=np.array(bbox1)
    bbox2=np.array(bbox2)
    y1_s,x1_s,y1_e,x1_e=bbox1.astype(np.float)
    y2_s,x2_s,y2_e,x2_e=bbox2.astype(np.float)
    str_y=max(y1_s-5,y2_s)
    stp_y=min(y1_e+5,y2_e)
    str_x=max(x1_s-5,x2_s)
    stp_x=min(x1_e+5,x2_e)
    x=stp_x-str_x
    y=stp_y-str_y
    if(x<=0 or y<=0):
        return [0,0]
    area_intersection=x*y
    area1=(y1_e-y1_s)*(x1_e-x1_s)
    area2=(y2_e-y2_s)*(x2_e-x2_s)

    area_union=area2+area1-area_intersection
    IoU=area_intersection/area_union
    IoBB=area_intersection/area2
    return [IoU,IoBB]


dirc='Final TEST M-S/Medium/' # Input Images Directory 
OutDir='Output/' # Directory to output image outputs
meta_file='DL_info.csv' # Link to meta file from Dataset
org_dir='Images/' # Link to folder containing origional files unsegmented 
b_box_dir='B_box/' # Link to folder containing files with bounding box created using Utilities/boudning_box.py

lung_low=90 # set lowest side size for selection of lung
lung_high=400 # set highest side size for selection of lung

save=False  # Set this to true to save the output

break_flag=False
info=pd.read_csv(meta_file)
lung_info=info[info.Coarse_lesion_type == 5] # 5 for lung lesions
f=open('classifier.pickle')
model=pickle.load(f)

false_positives=0
true_positives=0


#lesion bounding box
flag=False
counter=0
true_count=0
for ind,file in enumerate(os.listdir(dirc)):
    try:
        img=io.imread(dirc+file)
        org_img=io.imread(org_dir+file)
        b_box_img=io.imread(b_box_dir+file)
    except:
        continue
    print(file, ind)
    lesion=lung_info[lung_info.File_name==file]
    bboxes=[]
    for les in lesion.iterrows():
        les=les[1]
        scale=np.array(les['Spacing_mm_px_'].split(',')).astype(np.float)[0]
        b_box=np.array(les['Bounding_boxes'].split(',')).astype(np.float)
        b_box=b_box.astype(np.int)
        b_box=[b_box[1],b_box[0],b_box[3],b_box[2]]
        bboxes.append(b_box)
    #selection for lungs

    counter+=len(bboxes)
    thresh = threshold_otsu(img)
    bw=img
    file_parts=file.split('.')

    cleared = clear_border(bw)
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=img)
    
    new_img=np.zeros(bw.shape)
    
    select=[]
    min_diff=100000000
    min_diffr=100000000
    mindiffm=100000000
    for region in regionprops(label_image):
        
        # take regions with large enough areas
        if region.area >= 3000:
            select.append(region)
    first =-1
    second=-1
    for i in range(0,len(select)):
        minr1, minc, maxr1, maxc = select[i].bbox
        if(minr1 < lung_low or minr1 >lung_high):
            continue
        for j in range(0,len(select)):
            minr2, minc, maxr2, maxc = select[j].bbox
            if(minr2 < lung_low or minr2 >lung_high):
                continue    
            if(i!=j):
                #######################
                row_size=maxr1-minr1
                col_size=maxc-minc
                row_size=float(row_size)
                col_size=float(col_size)
                ratio1=row_size*col_size
                row_size=maxr1-minr2
                col_size=maxc-minc
                row_size=float(row_size)
                col_size=float(col_size)
                ratio2=row_size*col_size
                #######################
                diff=abs(select[i].area-select[j].area)
                diffR= abs(ratio1-ratio2)
                diffm= abs(minr1-minr2)
                if(diffR<min_diffr and diffm< mindiffm):
                    min_diff=diff
                    min_diffr=diffR
                    mindiffm = diffm 
                    first=i
                    second=j
    if(first>=0 and second>=0):
        select=[select[first],select[second]]
    elif(len(select)==0):
        continue
    elif(len(select)>1):
        select=[select[0],select[1]]
    else:
        select=[select[0]]
   ######################
    if(len(select)>1):
       minr1, minc1, maxr1, maxc1 = select[0].bbox
       minr2, minc2, maxr2, maxc2 = select[1].bbox
       if maxr1<minr2:
           select=[select[0]]
       elif minr1<minr2 and minc1 < minc2 and maxr2 < maxr1 and maxc2<maxc1:
           select=[select[1]]
       elif minr1>minr2 and minc1 > minc2 and maxr2 > maxr1 and maxc2>maxc1:
           select=[select[0]]      
  
    for region in select:
        minr, minc, maxr, maxc = region.bbox
        row_size=maxr-minr
        col_size=maxc-minc
        row_size=float(row_size)
        col_size=float(col_size)
        ratio=row_size/col_size
        for i in range(minr,maxr):
            for j in range(minc,maxc):
                if(region.convex_image[i-minr,j-minc] and not bw[i,j]):
                    new_img[i,j]=1
    
    
    # Filter out large connected chunks within lungs
    
    cleared = clear_border(new_img)
    cleared = erosion(cleared, square(2))
    
    labels_rw = label(cleared)
 
    if(save):    
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(b_box_img)
    number_true=0
    number_false=0
    rejected=0
    new_img=np.zeros(new_img.shape)
    bbox_pos=np.zeros(len(bboxes))
    for reg in regionprops(labels_rw,intensity_image=org_img):
        minr, minc, maxr, maxc = reg.bbox
        row_size=maxr-minr
        col_size=maxc-minc
        row_size=float(row_size)
        col_size=float(col_size)
        ratio=row_size/col_size
        bbx=row_size*col_size
        sum_i=float(0)
        nums=float(0)
        
        for i,row in enumerate(reg.convex_image):
            for j,col in enumerate(row):
                if(not col):
                    sum_i+=org_img[minr+i,minc+j]
                    nums+=1
        if(nums!=0):
            avg_near=sum_i/nums
        else:
            avg_near=0
        area=reg.area
        major_axis_length=reg.major_axis_length
        minor_axis_length=reg.minor_axis_length
        mean_intensity=reg.mean_intensity
        diameter=reg.equivalent_diameter*scale
        if(mean_intensity!=0 and avg_near!=0):
            ratio_intensity=mean_intensity/avg_near
        else:
            ratio_intensity=0
        feature=[area,ratio,reg.eccentricity,reg.euler_number,reg.extent,major_axis_length,minor_axis_length,mean_intensity,reg.min_intensity,reg.max_intensity,ratio_intensity]
        feature=np.expand_dims(np.array(feature), axis=0)
        
        if(model.predict(feature)==1):
            if(save):
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                 fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
            for b_ind,b_box in enumerate(bboxes):
                IoU,IoBB=eval_box(reg.bbox,b_box) # Here you get the Intersection over Union and Intersection over Bounding Box
                if(IoBB>=.3):                    # change the parameter to use for selection of true positives 
                    if(bbox_pos[b_ind]==1):
                        continue
                    bbox_pos[b_ind]+=1
                    true_positives+=1
                    number_true+=1
                    
                else:
                    false_positives+=1
                    number_false+=1
                    
        else:
            rejected+=1
        if(break_flag):
            break
    if(break_flag):
        break

    print("number_true",number_true)
    print("number_false",number_false)
    print("rejected",rejected)
    if(save):
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(OutDir+file_parts[0]+'_G'+'.png') 
        plt.clf()
        plt.close()

total=counter
print(counter)
true_positives=float(true_positives)
false_positives=float(false_positives)
total=float(total)
print('sensitivity:',true_positives, true_positives/total)
print('false_positives/image:',false_positives/total)



