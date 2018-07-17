# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 18:39:48 2018

@author: Student
"""

import os
import glob

from scipy.misc import imread,toimage
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import extract_patches_2d as ext_p
from skimage.feature import corner_harris as hc

def colour_grey(im1):
    size = im1.shape
    im = np.zeros((size[0],size[1]))
    for i in range(0,size[0]-1):
        for j in range(0,size[1]-1):
            if(len(size)==3):
                im[i][j] = 0.299*im1[i][j][0] + 0.587*im1[i][j][1] + 0.114*im1[i][j][2]
            else:
                im[i][j] = im1[i][j]
    return im

def getdescriptor(data,descriptor,no_int_point_array):
    for im_no in range(0,len(data)):
        im1 = data[im_no]
        size = im1.shape
        im = colour_grey(im1)
        k=0
        interest_point =[]
        for i in range(25,size[0]-1,25):
            for j in range(25,size[1]-1,25):
                interest_point = np.append(interest_point,i)
                interest_point = np.append(interest_point,j)
                k=k+1
        no_int_point = k
        no_int_point_array.append(k)
        interest_point = np.reshape(interest_point,[no_int_point,2]) 
        patches = np.zeros((no_int_point,21,21))
        int_point_desc = np.zeros((no_int_point,81))
        for k in range(0,no_int_point):
            [i,j] = [np.int(interest_point[k][0]),np.int(interest_point[k][1])]
            for m in range(-10,10):
                for n in range(-10,10):
                    if(m+i<size[0] and n+j < size[1]):
                        patches[k][m+10][n+10] = im[m+i][n+j]
                        int_point_desc[k] = np.transpose(hog(patches[k],9,(7,7)))
                        
        int_point_desc = np.reshape(int_point_desc , [1,(no_int_point*81)])
        descriptor.append(int_point_desc)
    return descriptor,no_int_point_array

    
label=[]
image_dir = "D:/To desktop/Mtech/sem2/projects/bovw/101_ObjectCategories/elephant2"
data_path = os.path.join(image_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = imread(f1)
    data.append(img)
    label.append(0)
no_images_1 = len(data)
print("Starting descriptor")
descriptor = []
no_int_point_array=[]
descriptor , no_int_point_array = getdescriptor(data,descriptor,no_int_point_array)
print("Next")    
image_dir = "D:/To desktop/Mtech/sem2/projects/bovw/101_ObjectCategories/face1"
data_path = os.path.join(image_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = imread(f1)
    data.append(img)
    label.append(1)
no_images_2 = len(data)
print("Next")
descriptor , no_int_point_array = getdescriptor(data,descriptor,no_int_point_array)
image_dir = "D:/To desktop/Mtech/sem2/projects/bovw/101_ObjectCategories/cellphone"
data_path = os.path.join(image_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = imread(f1)
    data.append(img)
    label.append(2)
no_images_3 = len(data)
print("Next")
descriptor , no_int_point_array = getdescriptor(data,descriptor,no_int_point_array)

print("Descriptor collection over")
desc1=[]
a=[]
for i in range(len(descriptor)):
    a=descriptor[i]
    for j in range(len(a)):
        desc1=np.append(desc1,a[j],axis=0)  # The concatenated descriptor
        
desc1 = np.asarray([desc1])
desc1 = desc1[0]
desc1 = np.reshape(desc1,[np.int(desc1.shape[0]/81) , 81])
kmeans = KMeans(30)
kmeans = kmeans.fit(desc1)
cluster_centers1 = kmeans.cluster_centers_ # it is 8x81
           
# Histogram ------------------- we have 15 cluster centers or in other words, the length of our visual dictionary = 15
dist = np.zeros((len(cluster_centers1) , len(descriptor) , np.amax(no_int_point_array)))
for i in range(len(cluster_centers1)):
    a=cluster_centers1[i]
    for j in range(len(descriptor)):
        b = descriptor[j]
        no_points = b.shape[1]/81
        for k in range(0,np.int(no_points)):
            c=np.array(b[0][k*81:(k+1)*81])
            if(k==0):
                dist[i][j][0] = np.linalg.norm(c-a)
            else:
                dist[i][j][k%81] = np.linalg.norm(c-a)
    d = np.argmax(dist[i])
histogram = np.zeros((len(descriptor) , len(cluster_centers1)))
for j in range(len(descriptor)):
    no_points = descriptor[j].shape[1]/81
    for k in range(0,np.int(no_points)):
        starting = dist[0][j][k]
        starting_cluster = 0
        for i in range(1,len(cluster_centers1)):
            if(dist[i][j][k] < starting):
                starting = dist[i][j][k]
                starting_cluster = i
                print(i)
        histogram[j][i] = histogram[j][i] + 1
lin_clf = svm.SVC()
lin_clf.fit(histogram,label)


# testing with car images--------------------------------
print("Testing starts")
image_dir = "D:/To desktop/Mtech/sem2/projects/bovw/101_ObjectCategories/test2"
data_path = os.path.join(image_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = imread(f1)
    data.append(img)
    label.append(0)
no_images_1 = len(data)       
print("Starting descriptor")
descriptor_test = []
no_int_point_array_test=[]
for im_no in range(0,len(data)):
    im1 = data[im_no]

#----------Dense interest point detection-------------------------------------
    size = im1.shape
    im = np.zeros((size[0],size[1]))
    for i in range(0,size[0]-1):
        for j in range(0,size[1]-1):
            if(len(size)==3):
                im[i][j] = 0.299*im1[i][j][0] + 0.587*im1[i][j][1] + 0.114*im1[i][j][2]
            else:
                im[i][j] = im1[i][j]

# Take every 15th point to be an interest point----------a patch size of 9x9 aroung that point
    k=0
    interest_point_test =[]
    for i in range(25,size[0]-1,25):
        for j in range(25,size[1]-1,25):
            interest_point_test = np.append(interest_point_test,i)
            interest_point_test = np.append(interest_point_test,j)
            k=k+1
    no_int_point_test = k
    no_int_point_array_test.append(k)
    #print(k)
    interest_point_test = np.reshape(interest_point_test,[no_int_point_test,2])   
#Drawing patches around the interest points-and getting feature descriptors using hog----------------------------------
#patch size=9x9
    patches_test = np.zeros((no_int_point_test,21,21))
    int_point_desc_test = np.zeros((no_int_point_test,81))
    for k in range(0,no_int_point_test):
        [i,j] = [np.int(interest_point_test[k][0]),np.int(interest_point_test[k][1])]
        for m in range(-10,10):
            for n in range(-10,10):
                if(m+i<size[0] and n+j < size[1] and m+10<21 and n+10<21):
                    patches_test[k][m+10][n+10] = im[m+i][n+j]
                    int_point_desc_test[k] = np.transpose(hog(patches_test[k],9,(7,7)))
                    
    int_point_desc_test = np.reshape(int_point_desc_test , [1,(no_int_point_test*81)])
    descriptor_test.append(int_point_desc_test)

dist_test = np.zeros((len(cluster_centers1) , len(descriptor_test) , np.amax(no_int_point_array_test)))
for i in range(len(cluster_centers1)):
    a=cluster_centers1[i]
    for j in range(len(descriptor_test)):
        b = descriptor_test[j]
        no_points = b.shape[1]/81
        for k in range(0,np.int(no_points)):
            c=np.array(b[0][k*81:(k+1)*81])
            if(k==0):
                dist_test[i][j][0] = np.linalg.norm(c-a)
            else:
                dist_test[i][j][k%81] = np.linalg.norm(c-a)
    d_test = np.argmax(dist_test[i])
histogram_test = np.zeros((len(descriptor_test) , len(cluster_centers1)))
for j in range(len(descriptor_test)):
    no_points = descriptor_test[j].shape[1]/81
    for k in range(0,np.int(no_points)):
        starting = dist_test[0][j][k]
        starting_cluster = 0
        for i in range(1,len(cluster_centers1)):
            if(dist_test[i][j][k] < starting):
                starting = dist_test[i][j][k]
                starting_cluster = i
                #print(i)
        histogram_test[j][i] = histogram_test[j][i] + 1
label_test = np.zeros((len(descriptor_test),1))
label_test = np.array([2,2,2,2,0,2,0,0,2,0,2,0,0,0,1,1,1,1,1,1,1])
tp = 0
fp = 0
right=[]
wrong=[]
misclass=[]
for i in range(0,21):
    for j in range(0,no_int_point_array_test[i]):
        x = lin_clf.predict(histogram_test[i])
        if(x == label_test[i]):
            
            tp = tp +1
        else:
            
            fp = fp +1
    if(tp>fp):
        right.append(i)
    else:
        print("Wrong Classification")
        wrong.append(i)
        misclass.append(x)
    tp=0
    fp=0
accuracy = tp/(tp+fp)
print("Accuracy for testing data = ")
print(accuracy*100)
im = imread("D:/To desktop/Mtech/sem2/projects/bovw/101_ObjectCategories/test2/image_0051.jpg")
plt.title('CellPhone')
plt.imshow(im)
