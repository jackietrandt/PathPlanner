import sys
import cv2.cv as cv

import kivy
kivy.require('1.0.6') # replace with your current kivy version !

from kivy.app import App
from kivy.uix.button import Button

import time

from math import sin, cos, sqrt, pi, fabs
import os
def PathFinderMain():
    #USB camera capture 
    capture1 = cv.CaptureFromCAM(0)
    capture2 = cv.CaptureFromCAM(1)

    #________Program configuration option______
    ModeTest = True # set capture frame resolution 768 x 1024 ___or 1080 x 1920
    CameraDebugScreen = True #show raw capture from camera 1 and 2 for alignment and checking ___or disable at run mode
    #Configure capture screen resolution
    if ModeTest:
        cv.SetCaptureProperty(capture1, cv.CV_CAP_PROP_FRAME_HEIGHT, 768)
        cv.SetCaptureProperty(capture1, cv.CV_CAP_PROP_FRAME_WIDTH, 1024)

        cv.SetCaptureProperty(capture2, cv.CV_CAP_PROP_FRAME_HEIGHT, 768)
        cv.SetCaptureProperty(capture2, cv.CV_CAP_PROP_FRAME_WIDTH, 1024)

    else:
        cv.SetCaptureProperty(capture1, cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        cv.SetCaptureProperty(capture1, cv.CV_CAP_PROP_FRAME_WIDTH, 1920)

        cv.SetCaptureProperty(capture2, cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        cv.SetCaptureProperty(capture2, cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
    while True:
    
        #img = cv.QueryFrame(capture)
        img1 = cv.QueryFrame(capture1)
        img2 = cv.QueryFrame(capture2)
    
        if CameraDebugScreen:
            cv.ShowImage("usbCam_1", img1)
            cv.ShowImage("usbCam_2", img2)
        if cv.WaitKey(2) == 27:
            break
        
class MyApp(App):

    def build(self):
        PathFinderMain()
        return Button(text='Hello World')

if __name__ == '__main__':
    MyApp().run()
            