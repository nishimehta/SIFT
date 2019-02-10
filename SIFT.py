#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:18:49 2018

@author: nishimehta
#person 50291671
#ubit nishimeh
"""


# coding: utf-8

# In[3]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:11:07 2018

@author: nishimehta
"""
import cv2
import numpy
import math

#evaluates the value of gaussian function
def evaluate(x,y,sig):
    return (math.exp(-(x**2+y**2)/(2*sig**2)))/(2*math.pi*sig**2)

#forms a gaussian matrix with dimesnsion n and sigma as sig
def form_gaussian_matrix(n,sig):
    gaussian=[[0 for x in range(n)] for y in range(n)]
    sum=0
    for i in range(n):
        for j in range(n):
            gaussian[i][j]=evaluate(i-int(n/2),j-int(n/2),sig)
            sum+=gaussian[i][j]
    #used for normalizing the kernel
    c=1/sum
    for i in range(n):
        for j in range(n):
            gaussian[i][j]*=c
    return gaussian     


# In[4]:
#pads image with n rows and columns with value as 0
def image_padding(im,n):
    
    for j in range(int(n/2)):
        for i in range(len(im)):
            im[i].insert(0,0)
    for j in range(int(n/2)):
        for i in range(len(im)):
            im[i].insert(len(im[i]),0)
            
    for i in range(int(n/2)):
        zeroes_list = [0 for x in range(len(im[0]))]
        im.insert(len(im),zeroes_list)
    for i in range(int(n/2)):
        zeroes_list = [0 for x in range(len(im[0]))]
        im.insert(0,zeroes_list)
    return im
#%% 
# extracts matrix of size nxn from matrix mat around position i,j
def extract_matrix(i,j,mat,n):
    x = int(n/2)
    extract=[[0 for x in range(n)] for y in range(n)] 
    for k in range(n):
        for l in range(n):
            extract[k][l] = mat[i-x+k][j-x+l]
    return extract
#%%
# performs convolution between matrix m and k
def convolution(m,k):
    product=0
    n=len(m)-1
    for i in range(len(m)):
        for j in range(len(m)):
            product+=(m[i][j] * k[(n-i)][n-j])
    return product
# In[5]:
#applies the gaussian kernel to the image
def apply_gaussian_kernel(im,g):
    # ignores boundary values
    new_image=[[0 for x in range(len(im[0]))] for y in range(len(im))] 
    for i in range(int(len(g)/2),len(im)-int(len(g)/2)):
        for j in range(int(len(g)/2),len(im[0])-int(len(g)/2)):
            # extracts matrix for each pixel
            m = extract_matrix(i,j,im,len(g))
            # result is the convolution
            new_image[i][j] = convolution(m,g)
    return new_image

# In[6]:
# generates n octaves of the image
def octave_generation(im,n):
    images=[]
    images.append(im)
    for k in range(n-1):
        x=pow(2,k+1)
        temp_image=[[0 for x in range(int(len(images[0][0])/x))] for y in range(int(len(images[0])/x))]
        l=0
        for i in range(0,len(im)-x,x):
            m=0
            for j in range(0,len(im[0])-x,x):
                temp_image[l][m] = im[i][j]
                m+=1
            l+=1
        images.append(temp_image)
    for i in range(len(images)):
        y=numpy.asarray(images[i])
        cv2.imwrite('octave'+str(i+1)+'.png',y)
        print(len(images[i]),len(images[i][0]))
    return images
# In[7]:
#applies the respective gaussian function to all the respective images 
def apply_gaussian_all(images,gaussian):
    gaussian_filtered_images=[]
    for i in range(len(gaussian)):
        temp=[]
        image_padded = image_padding(images[i],len(gaussian))
        for j in range(len(gaussian[0])):
            image_g = apply_gaussian_kernel(image_padded,gaussian[i][j])
            temp.append(image_g)
            y=numpy.asarray(image_g)
            cv2.imwrite('gaussian'+str(i)+'_'+str(j)+'.png',y)
        gaussian_filtered_images.append(temp)
    return gaussian_filtered_images

# In[8]:
#Computes the difference of gaussian for the images after gaussian blurring
def compute_DoG(gaussian_filtered_images):
    DoG=[]
    for i in range(len(gaussian_filtered_images)):
        temp=[]
        for j in range(len(gaussian_filtered_images[0])-1):
            temp_image = numpy.asarray(gaussian_filtered_images[i][j]) - numpy.asarray(gaussian_filtered_images[i][j+1])
            temp.append(temp_image.tolist())
            y = numpy.asarray(temp_image)
            cv2.imwrite('DoG'+str(i)+'_'+str(j)+'.png',y)
        DoG.append(temp)
    return DoG
# In[9]:
im = cv2.imread('task2.jpg',0)
image_list=im.tolist()
images = octave_generation(image_list,4)
        
# In[10]:

#value of sigmas provided in the question
sigmas = [[1/math.pow(2,1/2),1,math.pow(2,1/2),2,2*math.pow(2,1/2)],
           [math.pow(2,1/2),2,2*math.pow(2,1/2),4,4*math.pow(2,1/2)],
           [2*math.pow(2,1/2),4,4*math.pow(2,1/2),8,8*math.pow(2,1/2)],
           [4*math.pow(2,1/2),8,8*math.pow(2,1/2),16,16*math.pow(2,1/2)]]
gaussian=[]

#generates gaussian kernels
for i in range(len(sigmas)):
    temp=[]
    for j in range(len(sigmas[0])):
        temp.append(form_gaussian_matrix(7,sigmas[i][j]))
    gaussian.append(temp)
    


# In[16]:
#applies the respective gaussian kernels to the corresponding images
images_g = apply_gaussian_all(images,gaussian)



# In[17]:

DoG = compute_DoG(images_g)
# In[ ]:
im = cv2.imread('task2.jpg',0)
image_list = im.tolist()
for i in range(len(DoG)):

    octave=pow(2,i)
    x=octave*2
    
    for j in range(len(DoG[0])-2):

        stack=[]
        #to remove padding and stacks 3 DoGs together to find maxima and minima points of the image
        stack.append(DoG[i][j][3:-3][3:-3])
        stack.append(DoG[i][j+1][3:-3][3:-3])
        stack.append(DoG[i][j+2][3:-3][3:-3])
        for k in range(1,len(stack[0])-x-7):
            for l in range(1,len(stack[0][0])-x-7):
                minimum=1
                maximum=1
                value = stack[1][k][l]
                for m in range(3):
                    for n in range(-1,2):
                        for o in range(-1,2):
                            if(m==1 and n==o):
                                continue
                            if(stack[m][k+n][l+o]>value):
                                maximum=0
                            if(stack[m][k+n][l+o]<value):
                                minimum=0
                            if(abs(stack[m][k+n][l+o]-value)<0.2):
                                maximum=0
                                minimum=0
                if(minimum==1 or maximum==1):
                    if((k*octave)<75 and (l*octave)<75):                        
                    #mapping sub pixels to the original image
                    image_list[k*octave][l*octave]=255

y = numpy.asarray(image_list)
cv2.imwrite('result.png',y)

