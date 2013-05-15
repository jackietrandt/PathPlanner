import numpy as np
import cv2
import cv2.cv as cv
from time import clock
import sys

def Background_Sampler(img_background):
    hsv_map = np.zeros((180, 256, 3), np.uint8)
    h, s = np.indices(hsv_map.shape[:2])
    hsv_map[:,:,0] = h
    hsv_map[:,:,1] = s
    hsv_map[:,:,2] = 255
    hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)
    cv2.imshow('hsv_map', hsv_map)

    cv2.namedWindow('hist', 0)
    small = cv2.pyrDown(img_background)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    dark = hsv[...,2] < 32
    hsv[dark] = 0
    h = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    #hist scale from 10 to 32
    hist_scale = 10
    #np.clip(a, a_min, a_max) - clipping off histogram bin ?!
    h = np.clip(h*0.005*hist_scale, 0, 1)
    vis = hsv_map*h[:,:,np.newaxis] / 255.0
    cv2.imshow('hist', vis)
    
def Background_Trim(img_original,TrimParam):
    #if trim left right up, top bottom doesnt bigger than the whole image then
    height,width,depth = img_original.shape
    if (width - TrimParam['left'] - TrimParam['right'] > 0) & (height - TrimParam['top'] - TrimParam['bottom'] > 0):
        #img_trimmed = np.zeros((img_original.width - trim_left - trim_right, 
        #                        img_original.height - trim_top - trim_bottom, 
        #                        3), np.uint8)

        #then work out top left and bottom right 
        x1 = TrimParam['left']
        y1 = TrimParam['top']
        x2 = width - TrimParam['right']
        y2 = height - TrimParam['bottom']

        #then set roi
        img_trimmed = np.zeros((y2 - y1,x2 - x1,depth),np.uint8)
        img_trimmed[0:y2-y1,0:x2-x1] = img_original[y1:y2,x1:x2]
        
    return img_trimmed

#_____________________Used in________________________________________________
#__PathFinderMain():
#____________________________________________________________________________
#__Sample the 2 edge of the trimmed image, this then used for back projection
#__to getrid of the background and extract the object of interest
def Background_Sample(img_trimmed,BackgroundSample):
    height,width,depth = img_trimmed.shape
    #then work out if the Top border and Bottom border is not over lap each other and within range
    is_notoverlap = (height > (BackgroundSample['Top_border'] + BackgroundSample['Bottom_border']) )
    is_top_inrange = BackgroundSample['Top_border'] < height
    is_bottom_inrange = BackgroundSample['Bottom_border'] < height
    
    if (is_notoverlap & is_top_inrange & is_bottom_inrange):
        #if the parameter are all good then merge
        #create new image to hold the top and bottom sample bit
        img_top_sample = np.zeros((BackgroundSample['Top_border'],width,depth),np.uint8)
        img_bottom_sample = np.zeros((BackgroundSample['Bottom_border'],width,depth),np.uint8)
        img_merged_sample = np.zeros((BackgroundSample['Bottom_border'] + BackgroundSample['Top_border'],width,depth),np.uint8)
        #then we happy to copy it over
        img_top_sample[0:BackgroundSample['Top_border'],0:width] = img_trimmed[0:BackgroundSample['Top_border'],0:width]
        img_bottom_sample[0:BackgroundSample['Bottom_border'],0:width] = img_trimmed[height - BackgroundSample['Bottom_border']:height,0:width]
        #img_merged_sample [0:BackgroundSample['Top_border'],0:width] = img_trimmed[0:BackgroundSample['Top_border'],0:width]
        #img_merged_sample [BackgroundSample['Top_border']:BackgroundSample['Top_border'] + BackgroundSample['Bottom_border'],0:width] = img_trimmed[height - BackgroundSample['Bottom_border']:height,0:width]
        img_merged_sample = np.vstack((img_top_sample,img_bottom_sample))
        return img_merged_sample

#_____________________Used in________________________________________________
#__PathFinderMain():
#____________________________________________________________________________
#__Back projection , based on sampled background image, then calc histogram
#__and remove background bit which match the histogram, left the object of interest
def Background_remove(img_trimmed,sample_path):
    roi = cv2.imread(sample_path)
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)   
 
    target = img_trimmed
    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
 
    # calculating object histogram
    roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
 
    # normalize histogram and apply backprojection
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
 
    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)
 
    # threshold and binary AND
    ret,thresh = cv2.threshold(dst,5,255,0)
    #invert to get the object of interest
    cv2.bitwise_not(thresh,thresh)
    thresh = cv2.merge((thresh,thresh,thresh))
    res = cv2.bitwise_and(target,thresh)
 
    #res = np.vstack((target,thresh,res))
    return res

#_____________________Used in________________________________________________
#__
#____________________________________________________________________________
#__
#__
def Background_k_mean(img_trimmed):
    Z = img_trimmed.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center = cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img_trimmed.shape))
    return res2

#_____________________Used in________________________________________________
#__PathFinderMain():
#____________________________________________________________________________
#__Replace the cv2.imshow() function - which added debug switch
#__If the debug option is off then do not show all the image
def imshow(name,imgage,debug):
    if debug:
        cv2.imshow(name, imgage)
        
#_____________________Used in________________________________________________
#__PathFinderMain():
#____________________________________________________________________________
#__Extract object of interested base on detected bounding box from original image
#__
def Background_extract_obj(img_original,Box_hw):
    if Box_hw['valid']:
        height,width,depth = img_original.shape
        img_object = np.zeros((Box_hw['h'],Box_hw['w'],depth),np.uint8)
        img_object[0:Box_hw['h'],0:Box_hw['w']] = img_original[Box_hw['y']:Box_hw['y'] + Box_hw['h'],Box_hw['x']:Box_hw['x'] + Box_hw['w']]
    return img_object
