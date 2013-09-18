import sys
import cv2
import cv2.cv as cv
import numpy as np
from numpy import matrix
import threading
import Background as bg
#for scanning com port
import serial

import math
import time
from time import gmtime, strftime
from scipy import ndimage
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import QT_VERSION_STR

#for feature tracking
from common import getsize, draw_keypoints
from plane_tracker import PlaneTracker

#This for plotting histogram / print out matrix debug info and such
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
import pprint
import timeit
#queue for threading passing item between thread safely
import Queue
#machine learning 
#from sklearn import cluster, datasets
from math import sqrt

#---------------------------------------------------------------------------# 
# modbus serial com 
#---------------------------------------------------------------------------# 
from pymodbus.client.sync import ModbusSerialClient as ModbusClient
import time
import logging



# Class - Modbus class and object
#_____________________________________________________________________________
class Com_Modbus:
    def __init__(self):
        logging.basicConfig()
        log = logging.getLogger()
        log.setLevel(logging.DEBUG)
        
        #scan all available com port
        port_list = self.scan()
        print "Found ports:"
        for n,s in port_list: print "____(%d) %s" % (n,s)
        
        self.client = ModbusClient(method='ascii', port=s, baudrate='115200', timeout=1)
        print "Init Modbus Comm = ",self.client
        pass
    
    #Scan all available com port on this machine
    def scan(self):
       # scan for available ports. return a list of tuples (num, name)
        available = []
        for i in range(256):
            try:
                s = serial.Serial(i)
                available.append( (i, s.portstr))
                s.close()
            except serial.SerialException:
                pass

        return available

    def D_AddressRef(self,d_Address):
    #---------------------------------------------------------------------------# 
    # D_AddressRef
    #---------------------------------------------------------------------------# 
    #Input address in decimal - then function would convert it to PLC address
    #Address 0x1000 stand for D register in PLC
    #Address 0x0258 stand for 600 in decimal
    #So to write to D600 register in the PLC
    #The reference address is 0x1258
        d_Working = 4096
        d_Working = d_Working + d_Address
        return d_Working
    #_____________________________________________________________________________#
    # Usage example
    #_____________________________________________________________________________#
    #client.write_register(D_AddressRef(600), 123, unit=2 ) #unit=2 : mean PLC server address = 2
    #    def write_register(self, address, value, **kwargs): // Extracted from pymodbus\client\common.py
    #        '''
    #        :param address: The starting address to write to
    #        :param value: The value to write to the specified address
    #        :param unit: The slave unit this request is targeting
    #        :returns: A deferred response handle


# Class - Application and core functionality
#_____________________________________________________________________________

class App:
    def __init__(self, param):
        self.init_debug_facility()
        self.init_general_variable()
        self.init_threading_variable()
        #USB camera capture 
        
        self.capture2 = cv2.VideoCapture(0)
        self.capture1 = cv2.VideoCapture(1)
        print 'Camera 1 is openned = ',self.capture1.isOpened()
        print 'Camera 2 is openned = ',self.capture2.isOpened()
    def init_debug_facility(self):
        #________Program configuration option______
        self.ModeTest = True # set capture frame resolution 768 x 1024 ___or 1080 x 1920
        self.CameraDebugScreen = True #show raw capture from camera 1 and 2 for alignment and checking ___or disable at run mode
        self.ProcessDebug = False #show processed image in the middle
    def init_general_variable(self):
        #Share generic dictionary which used across functions
        self.KeyDictionary = {'key_up':2490368,
                         'key_down':2621440,
                         'key_left':2424832,
                         'key_right':2555904,
                         'key_r':114, #reset trim parameter
                         'key_d':100, #debug show image
                         'key_s':115, #sample background image
                         'key_t':116, #
                         'key_f':102, #rotate clock wise cam 1
                         'key_g':103, #rotate anti clock wise cam 1
                         'key_h':104, #rotate clock wise cam 2
                         'key_j':106, #rotate anti clock wise cam 2
                         'key_k':107, #trim left more cam 2
                         'key_l':108, #trim left less cam 2
                         'key_v':118, #virtical up cam 2
                         'key_b':98,  #virtical down cam 2
                         'key_n':110,
                         'key_m':109,
                         'key_ctr_s':19, #Save trim parameter
                         'key_ctr_r':18, #Reset cam 1 2 rotate and offset parameter
                         'key_tab':9, #move between operational mode
                         'key_esc':27 #stop all program and terminate thread 
                         }
        #Generic box for object of interest detection
        self.Box_hw = {'x':0,
                  'y':0,
                  'w':0,
                  'h':0,
                  'valid':False,
                  'w_old':0,
                  'h_old':0,
                  'variation':3
                  }
        #generic color code for opencv
        self.Color = {'red':(0,0,255),
                 'black':(0,0,0),
                 'green':(0,255,0),
                 'blue':(255,0,0),
                 'yellow':(0,255,255),
                 'white':(255,255,255)
                 }
    #Generic box xy
    def init_threading_variable(self):
        #hold list of executing thread
        self.threads = []
        self.event = threading.Event()
        #holding queue for passing object across thread
        self.queueLock_work = threading.Lock()
        self.queueLock_result = threading.Lock()
        self.workQueue = Queue.Queue(20)
        self.resultQueue = Queue.Queue(10)
        pass
    Box_xy = {'x1':0,
           'y1':0,
           'x2':0,
           'y2':0
          }
    #_____________________Used in________________________________________________
    #__class Configuration(object):
    #__Background_Trim(img_original,TrimParam):
    #____________________________________________________________________________
    #__Area which are trimmed out and not processed
    #__
    TrimParam = {'left':0,
                 'right':0,
                 'top':0,
                 'bottom':0,
                 'angle_left':0,
                 'angle_right':0,
                 'trim_left_left':0,
                 'off_virtical_left':0
                }
    #_____________________Used in________________________________________________
    #__class Configuration(object):
    #____________________________________________________________________________
    #__Area which will be sampled as background and
    #__used for backprojection, background sample 
    #__will be carried out every now and then
    #__
    BackgroundSample = {'Box_top':{},
                        'Box_bottom':{},
                        'Top_border':0,
                        'Bottom_border':0
                        }
    BackgroundSample['Box_top'] = Box_xy
    BackgroundSample['Box_bottom'] = Box_xy
    #_____________________Used in________________________________________________
    #__class Configuration(object):
    #____________________________________________________________________________
    #__Cutting dimension of  saw blade width, strip 1 width, saw blade width, strip 2 width, saw blade width
    #__used on x coordination histogram to figure out which cut path to take
    #__All dimension is in 1 mm
    CutParam = {'Blade_1':1,            #Top blade
                'Strip_1':10,           #Top strip
                'Blade_2':1,            #Middle blade
                'Strip_2':15,           #Bottom strip
                'Blade_3':1,            #Bottom blade
                'Good_threadhold':2,    #Number of defect feature counted, lower than or equal to this to count as good piece
                'Perfect_threadhold':0, #Number of defect feature counted, lower than or equal to this to count as perfect piece
                'Initialised':False}    #Initalise function should be called only once, then turn this true and not call it anymore
    #Configuration class - to hold initial startup configuration when project first load / run
    class Configuration(object):
        def __init__(self):
            #Define what in the config dictionary
            self.ConfigDictionary = {'TrimParam':{},
                                     'BackgroundSample':{},
                                     'CutParam':{}}
            #Define the structure of each config block
            self.ConfigDictionary['TrimParam'] = App.TrimParam
            self.ConfigDictionary['BackgroundSample'] = App.BackgroundSample
            self.ConfigDictionary['CutParam'] = App.CutParam
            #Define structure of run state
            self.KeySelectState = 'Operational','TrimArea','Background_top','Background_bottom'
            self.RunState = {'ConfigSaved':False,
                             'KeySelect': ['Operational','TrimArea','Background_top','Background_bottom'],
                             'KeySelectIndex': 0}
            #define some internal usage configuration variable
            self.Internal = {'Background_sample_path':r"Config\bg_sample.png"}
        #Save all the configuration block
        #this configuration then be save when press control S - and be load from file when program start to run
        def write(self):
            target = open('Config\config.ini', 'w')
            target.write(str(self.ConfigDictionary))
            print self.ConfigDictionary
        #Load all the configuratio block
        def read(self):
            try:
                s = open('Config\config.ini', 'r').read()
            except RuntimeError:
                print 'Config\config.ini _______ File not found'
            else:
                print 'Config\config.ini _______ File not found'
                #pass # does nothing
            if s <> '':
                self.ConfigDictionary = eval(s)
            else:
                print 'Config\config.ini _______ File is empty'
    
            print '_______________________________________________Load Configruation_______________________________________________'
            print self.ConfigDictionary
        #Update run state everytime a tab is pressed
        def RunStateUpdate(self):
            self.RunState['KeySelectIndex'] = self.RunState['KeySelectIndex'] + 1
            if (self.RunState['KeySelectIndex'] >= len(self.RunState['KeySelect'])):
                self.RunState['KeySelectIndex'] = 0
            print 'Run state = ',self.RunState['KeySelect'][self.RunState['KeySelectIndex']]
        def GetRunState(self):
            return self.RunState['KeySelect'][self.RunState['KeySelectIndex']]
    def KeyFunc_TrimArea(self,key,config):
    #_____________________Used in________________________________________________
    #__PathFinderMain():
    #____________________________________________________________________________
    #__Handle keyboard short cut to change trim area
    #__
        
        if key == self.KeyDictionary['key_up']:
            self.TrimParam['bottom'] = self.TrimParam['bottom'] + 2
        if key == self.KeyDictionary['key_down']:
            self.TrimParam['top'] = self.TrimParam['top'] + 2
        if key == self.KeyDictionary['key_left']:
            self.TrimParam['left'] = self.TrimParam['left'] + 2
        if key == self.KeyDictionary['key_right']:
            self.TrimParam['right'] = self.TrimParam['right'] + 2
        if key == self.KeyDictionary['key_r']:
            self.TrimParam['right'] = 0
            self.TrimParam['left'] = 0
            self.TrimParam['top'] = 0
            self.TrimParam['bottom'] = 0
        #Next section is for dual cam merge and rotate 
        if key == self.KeyDictionary['key_ctr_r']:
            self.TrimParam['angle_left'] = 0
            self.TrimParam['angle_right'] = 0
            self.TrimParam['trim_left_left'] = 0
            self.TrimParam['off_virtical_left'] = 0
        if key == self.KeyDictionary['key_f']:
            self.TrimParam['angle_left'] = self.TrimParam['angle_left'] + 0.1
        if key == self.KeyDictionary['key_g']:
            if self.TrimParam['angle_left'] > 0:
                self.TrimParam['angle_left'] = self.TrimParam['angle_left'] - 0.1
        #----
        if key == self.KeyDictionary['key_h']:
            self.TrimParam['angle_right'] = self.TrimParam['angle_right'] + 0.1
        if key == self.KeyDictionary['key_j']:
            if self.TrimParam['angle_right'] > 0:
                self.TrimParam['angle_right'] = self.TrimParam['angle_right'] - 0.1
        #----
        if key == self.KeyDictionary['key_k']:
            self.TrimParam['trim_left_left'] = self.TrimParam['trim_left_left'] + 1
        if key == self.KeyDictionary['key_l']:
            if self.TrimParam['trim_left_left'] > 0:
                self.TrimParam['trim_left_left'] = self.TrimParam['trim_left_left'] - 1
        #-----
        if key == self.KeyDictionary['key_v']:
            self.TrimParam['off_virtical_left'] = self.TrimParam['off_virtical_left'] + 1
        if key == self.KeyDictionary['key_b']:
            if self.TrimParam['off_virtical_left'] > 0:
                self.TrimParam['off_virtical_left'] = self.TrimParam['off_virtical_left'] - 1
    def KeyFunc_Background_top(self,key,config,BackgroundSample):
    #_____________________Used in________________________________________________
    #__PathFinderMain():
    #____________________________________________________________________________
    #__Handle keyboard short cut to change background sample top
    #__
        if key == self.KeyDictionary['key_down']:
            BackgroundSample['Top_border'] = BackgroundSample['Top_border'] + 2
        if key == self.KeyDictionary['key_up']:
            BackgroundSample['Top_border'] = BackgroundSample['Top_border'] - 2

    def KeyFunc_Background_bottom(self,key,config,BackgroundSample):
    #_____________________Used in________________________________________________
    #__PathFinderMain():
    #____________________________________________________________________________
    #__Handle keyboard short cut to change background sample bottom
    #__
        if key == self.KeyDictionary['key_down']:
            BackgroundSample['Bottom_border'] = BackgroundSample['Bottom_border'] - 2
        if key == self.KeyDictionary['key_up']:
            BackgroundSample['Bottom_border'] = BackgroundSample['Bottom_border'] + 2

    def Illustrate(self,image,position):
    #_____________________Used in________________________________________________
    #__PathFinderMain():
    #____________________________________________________________________________
    #__Illustrate the cut path on original image
    #__
        height, width, depth = image.shape
        offset = position + self.CutParam['Blade_1']
        pt1 = (0,offset)
        pt2 = (width,offset)
        cv2.line(image, pt1, pt2, self.Color['yellow'],1)     
        offset = offset + self.CutParam['Strip_1']
        pt1 = (0,offset)
        pt2 = (width,offset)
        cv2.line(image, pt1, pt2, self.Color['yellow'],1)     
        offset = offset + self.CutParam['Blade_2']
        pt1 = (0,offset)
        pt2 = (width,offset)
        cv2.line(image, pt1, pt2, self.Color['yellow'],1)     
        offset = offset + self.CutParam['Strip_2']
        pt1 = (0,offset)
        pt2 = (width,offset)
        cv2.line(image, pt1, pt2, self.Color['yellow'],1)     

    def MyFilledCircle(self,img, center, radius):
    #_____________________Used in________________________________________________
    #__Not yet
    #____________________________________________________________________________
    #__Filled circle draw to use in histogram sampling
    #__
        
        thickness = -1
        lineType = 8
        cv2.circle(img, center, radius, self.Color['white'], thickness, lineType) 

    def MyDraw_keypoints(self,vis, keypoints, radius):
    #_____________________Used in________________________________________________
    #__PathFinderMain():
    #____________________________________________________________________________
    #__Filled circle draw to use in histogram sampling
    #__
        color = self.Color['white']
        for kp in keypoints:
                x, y = kp.pt
                cv2.circle(vis, (int(x), int(y)), radius, color,-1,8)

    def Init_CutBand(self):
        self.cut_band_pair = np.arange(self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2'] + self.CutParam['Strip_2'] + self.CutParam['Blade_3'])
        for i in range(0, len(self.cut_band_pair)):
            if i < self.CutParam['Blade_1']:
                self.cut_band_pair[i] = 0
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1']):
                self.cut_band_pair[i] = 1
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2']):
                self.cut_band_pair[i] = 0
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2'] + self.CutParam['Strip_2']):
                self.cut_band_pair[i] = 1
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2'] + self.CutParam['Strip_2'] + self.CutParam['Blade_3']):
                self.cut_band_pair[i] = 0
            
        self.cut_band_1 = np.arange(self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2'] + self.CutParam['Strip_2'] + self.CutParam['Blade_3'])
        for i in range(0, len(self.cut_band_1)):
            if i < self.CutParam['Blade_1']:
                self.cut_band_1[i] = 0
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1']):
                self.cut_band_1[i] = 1
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2']):
                self.cut_band_1[i] = 0
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2'] + self.CutParam['Strip_2']):
                self.cut_band_1[i] = 0
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2'] + self.CutParam['Strip_2'] + self.CutParam['Blade_3']):
                self.cut_band_1[i] = 0

        self.cut_band_2 = np.arange(self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2'] + self.CutParam['Strip_2'] + self.CutParam['Blade_3'])
        for i in range(0, len(self.cut_band_2)):
            if i < self.CutParam['Blade_1']:
                self.cut_band_2[i] = 0
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1']):
                self.cut_band_2[i] = 0
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2']):
                self.cut_band_2[i] = 0
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2'] + self.CutParam['Strip_2']):
                self.cut_band_2[i] = 1
            elif i < (self.CutParam['Blade_1'] + self.CutParam['Strip_1'] + self.CutParam['Blade_2'] + self.CutParam['Strip_2'] + self.CutParam['Blade_3']):
                self.cut_band_2[i] = 0
    
    #machine learning helper function
    """This is a simple algorithm implemented in python that check whether or not 
        a value is too far (in terms of standard deviation) from the mean of a cluster"""
        
    def stat(self,lst):
        """Calculate mean and std deviation from the input list."""
        n = float(len(lst))
        mean = sum(lst) / n
        stdev = sqrt((sum(x*x for x in lst) / n) - (mean * mean)) 
        return mean, stdev
    
    def parse(self,lst, n):
        cluster = []
        for i in lst:
            if len(cluster) <= 1:    # the first two values are going directly in
                cluster.append(i)
                continue

            mean,stdev = self.stat(cluster)
            #if abs(mean - i) > n * stdev:    # check the "distance"
            if abs(mean - i) > (len(self.cut_band_pair)/1):    # check the "distance"
                yield cluster
                cluster[:] = []    # reset cluster to the empty list

            cluster.append(i)
        yield cluster           # yield the last cluster


    """This read image from 2 camera then merge them into 1 image, camera 0 will be on the left and 1 will be on the right """
    def merge_2cam(self):
        flas1,img1 = self.capture1.read()
        flas2,img2 = self.capture2.read()

        img2 = ndimage.rotate(img2, self.TrimParam['angle_right'])
        img1 = ndimage.rotate(img1, self.TrimParam['angle_left'])
   
        #__Merge__create a holder of the merged image from camera capture 1 and 2
        height, width, depth = img1.shape
        height2, width2, depth = img2.shape
        master_width = width*2 + 60
        img_Master = np.zeros((height + 60,master_width,depth),np.uint8)

        #__Merge__copy camera 0 ,1 image over master holder
        
        img_Master[0:height,0:width-10] = img1[0:height,0:width-10]
        img_Master[0:height2-self.TrimParam['off_virtical_left'],width-10:width-10+width2-self.TrimParam['trim_left_left']] = img2[self.TrimParam['off_virtical_left']:height2,self.TrimParam['trim_left_left']:width2]

        return img_Master

    def Camera_Monitoring(self):
        gain_old = self.capture1.get(cv2.cv.CV_CAP_PROP_GAIN)
        gain_old_2 = self.capture2.get(cv2.cv.CV_CAP_PROP_GAIN)
        #check if gain cap changed
        while True:
            time.sleep(10)
            """
            gain_new = self.capture1.get(cv2.cv.CV_CAP_PROP_GAIN)
            print 'CV_CAP_PROP_GAIN = ',self.capture1.get(cv2.cv.CV_CAP_PROP_GAIN)
            if gain_new <> gain_old:
                print 'CV_CAP_PROP_GAIN = ',self.capture1.get(cv2.cv.CV_CAP_PROP_GAIN)
                gain_old = gain_new
            """
    def PathFinderMain(self):
        config = self.Configuration()
        config.read()
        self.TrimParam = config.ConfigDictionary['TrimParam']
        BackgroundSample = config.ConfigDictionary['BackgroundSample']
        CutParam = config.ConfigDictionary['CutParam']
        CutParam['Initialised'] = False
        print '_______________________________________________Run State_______________________________________________'
        print config.RunState
        #Tracker for feature detection
        tracker = PlaneTracker()
    
        #Configure capture screen resolution
        if self.ModeTest:
           # self.capture1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1024)
           # self.capture1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 768)
           # self.capture2.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1024)
           # self.capture2.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 768)
            self.capture1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
            self.capture1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
            self.capture2.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
            self.capture2.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
        else:
            self.capture1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
            self.capture1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
            self.capture2.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
            self.capture2.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        self.Init_CutBand()
        #put it in trim mode first , to initialise other parameter before run
        config.RunStateUpdate()
        while True:

            img1 = self.merge_2cam()
            #flas,img2 = capture2.read()
            
            img_trimmed = bg.Background_Trim(img1,self.TrimParam)
            img_trimmed_overlay = img_trimmed.copy()
    
            if config.GetRunState() == 'Operational':
                #___________________________Smooth___________________________
                #img_trimmed = bg.Background_k_mean(img_trimmed)
                #img_trimmed = cv2.GaussianBlur(img_trimmed,(3,3),0)
                #img_trimmed = cv2.bilateralFilter(img_trimmed,3, 3*2,3/2)
                #_____________________________________________________________

                #___________________________Openning__________________________________
                #
                img_trimmed = bg.Background_Opening(img_trimmed, 10,2)
                #img_backprojected = bg.Background_k_mean(img_backprojected)
                bg.imshow("Open", img_trimmed,self.ProcessDebug)
                #_____________________________________________________________________
            
                #__________________________Draw Rect__________________________________
                #
                #Draw a black rectangular around the trimmed image to help with Canny close loop
                height, width, depth = img_trimmed.shape
                cv2.rectangle(img_trimmed, (0,0), (width,height), self.Color['black'],4)
                #_____________________________________________________________________
            
                #___________________________Back projection___________________________
                #Back project clear out the background using hist of sampled background image
                img_backprojected = bg.Background_remove(img_trimmed,config.Internal['Background_sample_path'])
                img_backprojected = bg.Background_k_mean(img_backprojected)
            
                bg.imshow("BackProject", img_backprojected,self.ProcessDebug)
                #img_backprojected = img_trimmed
                #_____________________________________________________________________
    
                #___________________________morphologyEx_______________________________
                #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                #res = cv2.morphologyEx(img_backprojected,cv2.MORPH_OPEN,kernel)
                #bg.imshow("morphologyEx", res,ProcessDebug)
                #______________________________________________________________________
            
                #___________________________Smooth___________________________
                #img_trimmed = bg.Background_k_mean(img_trimmed)
                #res = cv2.GaussianBlur(res,(3,3),0)
                #res = cv2.bilateralFilter(res,3, 3*2,3/2)
                #_____________________________________________________________        
            
                #___________________________Canny___________________________
                #img_trimmed = bg.Background_k_mean(img_trimmed)
                img_canny = cv2.Canny(img_backprojected, 80, 200)
                bg.imshow("Canny", img_canny,self.ProcessDebug)
                #___________________________________________________________
    
                #_________________________Erode__________________________________
                #4.Erodes the Thresholded Image
                #2.Converts to Gray level
                #cvtcolorImage = cv2.cvtColor(img_canny,cv2.cv.CV_RGB2GRAY)       
                #element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
                #cv2.erode(img_canny,element)
                #cv2.imshow('Eroded',img_canny)      
                #_______________________________________________________________
    
                #_________________________Dilate__________________________________
                kernel = np.ones((5,5),'int')
                img_canny = cv2.dilate(img_canny,kernel)
                bg.imshow("Dilate",img_canny,self.ProcessDebug)
                #_________________________________________________________________
            
                #___________________________Box it___________________________
                contours, hierarchy = cv2.findContours(img_canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                try:
                    cnt=contours[0]
                except IndexError:
                        print 'No object detected'
                else:
                    x,y,w,h = cv2.boundingRect(cnt)
                    if (w > 0) & (h > 0):
                        self.Box_hw['x'] = x
                        self.Box_hw['y'] = y
                        self.Box_hw['w'] = w
                        self.Box_hw['h'] = h
                        self.Box_hw['valid'] = True
                    else:
                        self.Box_hw['valid'] = False
                        print 'Box not found'
                    cv2.rectangle(img_backprojected,(x,y),(x+w,y+h),(0,255,0),2)
                    bg.imshow("Show_Contour",img_backprojected,self.ProcessDebug)
                #____________________________________________________________
            
                #___________________________Extract object____________________________
                #only update if box h w change more than an average variation
                img_object_of = np.zeros((self.Box_hw['h'],self.Box_hw['w'],depth),np.uint8)
                if  (True)|(math.fabs(self.Box_hw['w'] - self.Box_hw['w_old']) > self.Box_hw['variation'])|(math.fabs(self.Box_hw['h'] - self.Box_hw['h_old']) > self.Box_hw['variation']):
                
                    self.Box_hw['w_old'] = self.Box_hw['w']
                    self.Box_hw['h_old'] = self.Box_hw['h']
                    img_object_of = bg.Background_extract_obj(img_trimmed_overlay, self.Box_hw)
                bg.imshow ('object of interest',img_object_of,False) 
                
                img_object_of = img_trimmed_overlay
            
                #_____________________________________________________________________
                #img_trimmed_overlay_cv = cv.fromarray(img_trimmed_overlay)
    
                #___________________________Trim object of interest before threshold____________________________
                #Trim offset top and bottom of object of interest before threshold 
            
                TrimParam_of = {'left':0,
                                 'right':0,
                                 'top':3,
                                 'bottom':6
                                 }
            
                img_of_trimmed = bg.Background_Trim(img_object_of,TrimParam_of)
            
                #________________________________________________________________________________________________
            
                #___________________________Detect Feature____________________________
                #Detect feature on object of interest
                tracked = tracker.track(img_of_trimmed)        
                img_object_of_feature = img_of_trimmed.copy()
                draw_keypoints(img_object_of_feature, tracker.frame_points)

                bg.imshow ('feature',img_object_of_feature,False)
                #_____________________________________________________________________
     
                #___________________________Adaptive threshold____________________________
                #Detect feature on object of interest using adaptive threshold 
                img_object_of_adaptive = cv2.cvtColor(img_of_trimmed,cv2.cv.CV_RGB2GRAY)
                img_object_of_adaptive_blur = cv2.medianBlur(img_object_of_adaptive,5)

                th3 = cv2.adaptiveThreshold(img_object_of_adaptive_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,2)
                th3 = cv2.medianBlur(th3,3)
    
                bg.imshow ('feature_threadhold_adaptive_2',th3,self.ProcessDebug)
                #_____________________________________________________________________
            
                #___________________________Histogram of dark area____________________________
                height, width = th3.shape    
            
                #cv2.convertScaleAbs(th3,th3,1/255)
                dark_feature_hist = th3.sum(axis = 1)
                dark_feature_hist = np.divide(dark_feature_hist,255)
                #Plot result if need
                #plt.plot(dark_feature_hist)
                #plt.show()
                #There will be some noise from adaptive threadhold which need to be filter out
                noise_offset = 10
                list_offset = [noise_offset]*len(dark_feature_hist)
    
                a = matrix(dark_feature_hist)
                b = matrix(list_offset)
                ret = a - b
                #dark_feature_hist_offset = np.array(ret).reshape(-1,).tolist()
                #dark_feature_hist_offset = ret
                dark_feature_hist_offset = np.squeeze(np.asarray(ret))
                n = 0
                for i in dark_feature_hist_offset:
                    if (i <0):
                        dark_feature_hist_offset[n] = 0
                    n = n + 1
            
                #_____________________________________________________________________________
            
                #___________________________y coor histogram____________________________
                #Detect feature on object of interest
                #Then need to do a histogram of the detected feature point
                height, width, depth = img_object_of_feature.shape
    
                x_coor = []
                y_coor = []
                for kp in tracker.frame_points:
                    x, y = kp.pt
                    x_coor.append(int(x))
                    y_coor.append(int(y))
                #print y_coor
                #work out histogram 
                #n, bins, patches = plt.hist(y_coor,bins=height,range=(0,height),normed=False,weights=None,cumulative=False,bottom=None,histtype='bar',align='mid',orientation='vertical',rwidth=None,log=False,color=None,label=None,stacked=False)
                bins = np.arange(height+1)
                hist, bin_edges = np.histogram(y_coor, bins)
                #_______________________________________________________________________
            
                #__________________________________________combine 2 feature histogram_________________________________
                #combine 2 feature histogram - with weight scale for line search
            
                a = matrix(dark_feature_hist_offset)
                b = matrix(hist)
                ret = a*1 + b*1
                #dark_feature_hist_offset = np.array(ret).reshape(-1,).tolist()
                #dark_feature_hist_offset = ret
                hist_combine = np.squeeze(np.asarray(ret))
                #print hist_combine
                #______________________________________________________________________________________________________
            
                #______________________________________generate strip matrix__________________________

                array_and_sum = [0]*(len(hist_combine) - len(self.cut_band_pair))
                for i in range(len(hist_combine) - len(self.cut_band_pair)):
                    and_result = bg.Background_operator_and (self.cut_band_pair, hist_combine,i)
                    #print 'result = ',and_result 
                    array_and_sum[i] = and_result.sum()
                #_____________________________________________________________________________________
                
                
                #______________________________________Search cut line__________________________________________
                cut_list = []
                search_index = 0
                while(search_index < len(array_and_sum)):
                    #implement of binary search
                    if (array_and_sum[search_index] == 0):
                        #illustrate cut line and note down the first cut reference position in a list
                        cut_list.append(search_index)
                        
                        #Jump to next possible pair and start searching for a possible cut slot
                        search_index = search_index + len(self.cut_band_pair)
                    else:
                        search_index = search_index + 1
                #Append new cut list into thread queue for processing in machine learning thread
                self.queueLock_work.acquire()
                for cut_location in cut_list:
                    if not self.workQueue.full():
                        self.workQueue.put(cut_location)
                self.queueLock_work.release()
                
                #print out result monitoring
                print "_____________________Cut List_____________________"
                print cut_list
                #then we illustrate the cut slide 
                for i in cut_list:
                    #img_object_of_feature = Illustrate (img_object_of_feature,i)
                    self.Illustrate (img_object_of_feature,i)
                bg.imshow ('Illustrated',img_object_of_feature,True)
                #_______________________________________________________________________________________________
            else:
                cv2.imshow('Trimmed',img_trimmed)
            key = cv.WaitKey(2)
            #if key <> -1:
                #print key
            if key == self.KeyDictionary['key_esc']:
                self.event.set()
                break
            if key == self.KeyDictionary['key_d']:
                self.ProcessDebug = not self.ProcessDebug
                if self.ProcessDebug:
                    print 'Process Debug Screen = ', self.ProcessDebug
            #Let save the configuration to file
            if key == self.KeyDictionary['key_ctr_s']:
                config.ConfigDictionary['TrimParam'] = self.TrimParam
                config.ConfigDictionary['BackgroundSample'] = BackgroundSample 
                config.write()
            #Change KeySelect function mode
            if key == self.KeyDictionary['key_tab']:
                config.RunStateUpdate()
            #Let sample so background image
            if key == self.KeyDictionary['key_s']:
                img_background_sample = bg.Background_Sample(img_trimmed, BackgroundSample)
                cv2.imwrite(config.Internal['Background_sample_path'], img_background_sample) 
                cv2.imshow("Sampled", img_background_sample)
            if config.GetRunState() == 'TrimArea':
                self.KeyFunc_TrimArea(key,config)
            #Edit limit of background sample box on top
            if config.GetRunState() == 'Background_top':
                self.KeyFunc_Background_top(key,config,BackgroundSample)
                
                height,width,depth = img_trimmed_overlay.shape
                pt1 = (0,BackgroundSample['Top_border'])
                pt2 = (width,BackgroundSample['Top_border'])
                cv2.line(img_trimmed_overlay, pt1, pt2, self.Color['red'],2) 
            #Edit limit of background sample box on top
            if config.GetRunState() == 'Background_bottom':
                self.KeyFunc_Background_bottom(key,config,BackgroundSample)
                height,width,depth = img_trimmed_overlay.shape
                pt1 = (0,height -BackgroundSample['Bottom_border'])
                pt2 = (width,height - BackgroundSample['Bottom_border'])
                cv2.line(img_trimmed_overlay, pt1, pt2, self.Color['red'],2) 
            if self.CameraDebugScreen:
                bg.imshow ('Camera 1',img1,self.ProcessDebug)    
    def MachineLearning(self):
        while True:
            self.queueLock_work.acquire()
            #if we have enough sample in the queue then we use machine learning to seperate grouped cut
            if self.workQueue.full():
                data = []
                result = []
                while not self.workQueue.empty():
                    data.append(self.workQueue.get())
                data = sorted(data)
                print "__________________________________________Clustered sample __________________________________________"
                after_parse = self.parse(data, 7)
                for cluster in after_parse:
                    result.append(sum(cluster)/len(cluster))
                    print (cluster)
                print ('result = ',result)
            self.queueLock_work.release()
            """
            >>> import numpy as np
            >>> from scipy.cluster.vq import kmeans, vq
            >>> y = np.array([1,2,3,60,70,80,100,220,230,250])
            >>> codebook, _ = kmeans(y, 3)  # three clusters
            >>> cluster_indices, _ = vq(y, codebook)
            >>> cluster_indices
            array([1, 1, 1, 0, 0, 0, 0, 2, 2, 2])
            """
            time.sleep(1)
        

    def test(self):
        pass
    def run(self):
        pass
    def close_program(self):
        for each_thread in self.threads:
            each_thread.stop()
        self.capture1.release()
        self.capture2.release()
        del self.capture1
        del self.capture2
        
        pass


# Class - Text box for putting in note and accept / reject the selected area.
#_____________________________________________________________________________
class Dialog(QtGui.QDialog):
    def __init__(self):
        super(Dialog, self).__init__()

# Main program where everything start
#_____________________________________________________________________________
 
def main():

    print ("Python version = ",sys.version)
    print ("Opencv version = ",cv.__version__)
    print("Qt version = ", QT_VERSION_STR)
    print("Numpy version = ", np.version.version)
    print ("Matplotlib version = ",matplotlib.__version__)
    print ("Line profiler = ",matplotlib.__version__)
    param = True
    application = App(param)
    Modbus_Client = Com_Modbus()
    try:
        #Class for Camera Gain monitor
        class myThread_1 (threading.Thread):
            def __init__(self, threadID, name, counter):
                threading.Thread.__init__(self)
                self.threadID = threadID
                self.name = name
                self.counter = counter
            def run(self):
                print "Starting " + self.name
                application.Camera_Monitoring()
            def stop(self):
                self._Thread__stop()                
        #Class for PathFinder
        class myThread_2 (threading.Thread):
            def __init__(self, threadID, name, counter):
                threading.Thread.__init__(self)
                self.threadID = threadID
                self.name = name
                self.counter = counter
            def run(self):
                print "Starting " + self.name
                application.PathFinderMain()
            def stop(self):
                self._Thread__stop()                
        #Class for Machine learn to separate sample
        class myThread_3 (threading.Thread):
            def __init__(self, threadID, name, counter):
                threading.Thread.__init__(self)
                self.threadID = threadID
                self.name = name
                self.counter = counter
            def run(self):
                print "Starting " + self.name
                application.MachineLearning()
            def stop(self):
                self._Thread__stop()                
       
        # Create new threads
        thread1 = myThread_1(1, "Camera Monitor", 1)
        thread2 = myThread_2(2, "Path Finder", 2)
        thread3 = myThread_3(3, "Machine Learning", 3)
        
        # Start new Threads
        thread1.start()
        thread2.start()
        thread3.start()

        # Add threads to thread list
        application.threads.append(thread1)
        application.threads.append(thread2)
        application.threads.append(thread3)
        

        #thread.start_new_thread( application.Camera_Monitoring())
        #thread.start_new_thread( application.PathFinderMain())
        
    except:
        print "Error: unable to start thread"

    application.event.wait()
    application.close_program()
    
if __name__ == "__main__":
    main()