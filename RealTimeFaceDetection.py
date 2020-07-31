#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import matplotlib.pyplot as plt
num_of_sample = 52
vid = cv2.VideoCapture(0) # to open the camera
# haar cascade for frontal face
face_cascade = cv2.CascadeClassifier('//home//student//Desktop//haarcascade_frontalface_default.xml')
iter1=0
while(iter1<num_of_sample):
    r,frame = vid.read();# capture a single frame
    frame = cv2.resize(frame,(640,480)) # resizig the frame
    im1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# gray scale conversion of
    # color image
    face=face_cascade.detectMultiScale(im1)
    for x,y,w,h in (face):
        # [255,0,0] #[B,G,R] 0 to 255 
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),[0,0,255],4)
        iter1=iter1+1
        im_f = im1[y:y+h,x:x+w]
        im_f = cv2.resize(im_f,(112,92))#orl face matching size
        cv2.putText(frame,'sample no.'+str(iter1),(x,y), cv2.FONT_ITALIC, 1,
                   (255,0,255),2,cv2.LINE_AA)
        path2 = '/home/student/Desktop/photos/u1/%d.png'%(iter1) # path to save the image
        cv2.imwrite(path2,im_f) # to save the image 
        
    cv2.imshow('frame',frame)# display
    cv2.waitKey(1)
vid.release()

cv2.destroyAllWindows()


# In[9]:


import numpy as np 
import matplotlib.image as mimg
import matplotlib.pyplot as plt
from skimage import feature
from sklearn import svm
#from sklearn.externals import joblib
import pickle as p
train_data=np.zeros((7*41,280))
train_label=np.zeros((7*41))
count=-1
#plt.figure(1)
#plt.ion()
# feature extraction 
for i in range(1,3):
    for j in range(1,20):
        plt.cla()
        count=count+1
         
        path = '/home/student/Desktop/photos/u%d/%d.png'%(i,j)
        im = mimg.imread(path)
        feat,hog_image = feature.hog(im,orientations=8,pixels_per_cell=(16,16),
                                     visualize=True,block_norm='L2-Hys',
                                     cells_per_block=(1,1))
        train_data[count,:]=feat.reshape(1,-1)
        train_label[count]=i
        plt.subplot(2,1,1)
        plt.imshow(im,cmap='gray')
        plt.subplot(2,1,2)
        plt.imshow(hog_image,cmap='gray')
        plt.pause(0.1)
        print(i,j)

# model creation
svm_model = svm.SVC(kernel='poly',gamma='scale')

# train the model
svm_model = svm_model.fit(train_data,train_label)
f=open("//home//student//Desktop//svm_face_train_modelnew.pkl","wb")

p.dump(svm_model,f)

print('training done ')


# In[19]:


import cv2
import matplotlib.pyplot as plt
from skimage import feature
import pickle as p
f = open("//home//student//Desktop//svm_face_train_modelnew.pkl","rb")
svm_model=p.load(f)
num_of_sample = 52
vid = cv2.VideoCapture(0) # to open the camera
# haar cascade for frontal face
face_cascade = cv2.CascadeClassifier('//home//student//Desktop//haarcascade_frontalface_default.xml')
iter1=0
while(iter1<num_of_sample):
    r,frame = vid.read();# capture a single frame
    frame = cv2.resize(frame,(640,480)) # resizig the frame
    im1 = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)# gray scale conversion of
    # color image
    face=face_cascade.detectMultiScale(im1)
    for x,y,w,h in (face):
        # [255,0,0] #[B,G,R] 0 to 255 
        cv2.rectangle(frame,(x,y),(x+w,y+h),[0,0,255],4)
        iter1=iter1+1
        im_f = im1[y:y+h,x:x+w]
        im_f = cv2.resize(im_f,(112,92))#orl face matching size
        
        feat,hog_image = feature.hog(im_f,orientations=8,pixels_per_cell=(16,16),
                                     visualize=True,block_norm='L2-Hys',
                                     cells_per_block=(1,1))
        val1=svm_model.predict(feat.reshape(1,-1))
        str1=" "
        if val1[0]==1:
            str1="Abdul Kalam"
        else:
            str1="Others"
                
        cv2.putText(frame,str1,(x,y), cv2.FONT_ITALIC, 1,
                   (255,0,255),3,cv2.LINE_AA)
         
       
    cv2.imshow('frame',frame)# display
    cv2.waitKey(1)
    
vid.release()
cv2.destroyAllWindows()


# In[3]:


#EXAMPLE FOR EXTRACTING FEATURES FROM IMAGE
from skimage import  data, color, feature
image = color.rgb2gray(data.chelsea())
hogVec, hogVis = feature.hog(image, visualize=True)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')
ax[1].imshow(hogVis)
ax[1].set_title("extarcting features from image")
plt.show()


# In[ ]:




