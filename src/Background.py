import numpy as np
import cv2
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
    