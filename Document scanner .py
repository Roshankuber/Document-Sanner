#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


img_path="./bill2.jpg"
img = cv2.imread(img_path)
print(img.shape)
#converting bgr to rgb
img=cv2.resize(img,(900,800))
print(img.shape)


# In[3]:


plt.imshow(img)
plt.show()


# Remove the noise,
# Edge detection,
# Contour Extraction, 
# Best Contour selection,
# Project to the screen,

# In[4]:


#Remove the noise(img blurring)
org_img=img.copy()
gray=cv2.cvtColor(org_img,cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap="binary")
plt.show()


# In[5]:


#making img Blur
blurred = cv2.GaussianBlur(gray, (5,5), 0)
plt.imshow(blurred, cmap="binary")
plt.show()


# In[6]:


regen= cv2.cvtColor(blurred,cv2.COLOR_GRAY2BGR)
plt.imshow(org_img)
plt.show()

plt.imshow(regen)
plt.show()


# In[7]:


regen.shape


# In[8]:


#edge detection
edge=cv2.Canny(blurred,0,90)
org_edge=edge.copy()

plt.imshow(org_edge)
plt.title("Edge detection")
plt.show()


# In[9]:


#contours Extraction
contours, _ =cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
print("Number of contours:", len(contours))
contours = sorted(contours, key=cv2.contourArea, reverse=True)


# In[10]:


#select the best contours region 


# In[11]:


for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c,0.01*p,True)
    if len(approx)==4:
        target=approx
        break
        
print(target.shape)


# In[12]:


#reorder target contour
def reorder(h):
    h=h.reshape((4,2))
    print(h)
    hnew = np.zeros((4,2), dtype = np.float32)
    add=h.sum(axis=1)
    hnew[2] = h[np.argmax(add)]
    hnew[1] = h[np.argmax(add)]
    
    diff =np.diff(h, axis=1)
    hnew[0] = h[np.argmax(add)] 
    hnew[2] = h[np.argmax(add)]
    return hnew


# In[13]:


reorder = reorder(target)
print("::::::::::::::::")
print(reorder)


# In[14]:


input_representation = np.array(reorder)
output_map = np.float32([[0, 0], [700, 0], [700, 700], [0, 800]])

m = cv2.getPerspectiveTransform(input_representation, output_map)
ans = cv2.warpPerspective(org_img, m, (1000, 1000))


# In[15]:


plt.imshow(ans)
plt.show()


# In[16]:


res=cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY)
b_res = cv2.GaussianBlur(res,(3,3),0)
plt.imshow(res, cmap="gray")
plt.title("FINAL SCAN")
plt.show()


# In[ ]:




