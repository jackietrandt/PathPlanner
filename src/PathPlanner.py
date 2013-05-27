import sys
import cv2
import cv2.cv as cv
import numpy as np
import math
from time import gmtime, strftime

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

#________Program configuration option______
ModeTest = True # set capture frame resolution 768 x 1024 ___or 1080 x 1920
CameraDebugScreen = True #show raw capture from camera 1 and 2 for alignment and checking ___or disable at run mode
ProcessDebug = False #show processed image in the middle

#Share generic dictionary which used across functions
KeyDictionary = {'key_up':2490368,
                 'key_down':2621440,
                 'key_left':2424832,
                 'key_right':2555904,
                 'key_s':115,
                 'key_t':116,
                 'key_ctr_s':19,
                 'key_tab':9
                }

Box_xy = {'x1':0,
       'y1':0,
       'x2':0,
       'y2':0
      }

Box_hw = {'x':0,
          'y':0,
          'w':0,
          'h':0,
          'valid':False,
          'w_old':0,
          'h_old':0,
          'variation':3
          }

Color = {'red':(0,0,255),
         'black':(0,0,0),
         'green':(0,255,0),
         'blue':(255,0,0),
         'white':(255,255,255)
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
             'bottom':0
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
        self.ConfigDictionary['TrimParam'] = TrimParam
        self.ConfigDictionary['BackgroundSample'] = BackgroundSample
        self.ConfigDictionary['CutParam'] = CutParam
        
        #Define structure of run state
        self.KeySelectState = 'Operational','TrimArea','Background_top','Background_bottom'
        self.RunState = {'ConfigSaved':False,
                         'KeySelect': ['Operational','TrimArea','Background_top','Background_bottom'],
                         'KeySelectIndex': 0}
        
        #define some internal usage configuration variable
        self.Internal = {'Background_sample_path':r"Config\bg_sample.png"}
        
    #Save all the configuration block
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
    def RunStateUpdate(self):
        self.RunState['KeySelectIndex'] = self.RunState['KeySelectIndex'] + 1
        if (self.RunState['KeySelectIndex'] >= len(self.RunState['KeySelect'])):
            self.RunState['KeySelectIndex'] = 0
        print 'Run state = ',self.RunState['KeySelect'][self.RunState['KeySelectIndex']]
    def GetRunState(self):
        return self.RunState['KeySelect'][self.RunState['KeySelectIndex']]
#this configuration then be save when press control S - and be load from file when program start to run


# Background module which handle sampling background histogram / filtering
import Background as bg

#_____________________Used in________________________________________________
#__PathFinderMain():
#____________________________________________________________________________
#__Handle keyboard short cut to change trim area
#__
def KeyFunc_TrimArea(key,config,TrimParam):
    if key == KeyDictionary['key_up']:
        TrimParam['bottom'] = TrimParam['bottom'] + 2
    if key == KeyDictionary['key_down']:
        TrimParam['top'] = TrimParam['top'] + 2
    if key == KeyDictionary['key_left']:
        TrimParam['left'] = TrimParam['left'] + 2
    if key == KeyDictionary['key_right']:
        TrimParam['right'] = TrimParam['right'] + 2


#_____________________Used in________________________________________________
#__PathFinderMain():
#____________________________________________________________________________
#__Handle keyboard short cut to change background sample top
#__
def KeyFunc_Background_top(key,config,BackgroundSample):
    if key == KeyDictionary['key_down']:
        BackgroundSample['Top_border'] = BackgroundSample['Top_border'] + 2
    if key == KeyDictionary['key_up']:
        BackgroundSample['Top_border'] = BackgroundSample['Top_border'] - 2

#_____________________Used in________________________________________________
#__PathFinderMain():
#____________________________________________________________________________
#__Handle keyboard short cut to change background sample bottom
#__
def KeyFunc_Background_bottom(key,config,BackgroundSample):
    if key == KeyDictionary['key_down']:
        BackgroundSample['Bottom_border'] = BackgroundSample['Bottom_border'] - 2
    if key == KeyDictionary['key_up']:
        BackgroundSample['Bottom_border'] = BackgroundSample['Bottom_border'] + 2

#_____________________Used in________________________________________________
#__PathFinderMain():
#____________________________________________________________________________
#__Illustrate the cut path on original image
#__
def Illustrate (image,position):
    height, width, depth = image.shape
    offset = position + CutParam['Blade_1']
    pt1 = (0,offset)
    pt2 = (width,offset)
    cv2.line(image, pt1, pt2, Color['blue'],1)     
    offset = offset + CutParam['Strip_1']
    pt1 = (0,offset)
    pt2 = (width,offset)
    cv2.line(image, pt1, pt2, Color['blue'],1)     
    offset = offset + CutParam['Blade_2']
    pt1 = (0,offset)
    pt2 = (width,offset)
    cv2.line(image, pt1, pt2, Color['blue'],1)     
    offset = offset + CutParam['Strip_2']
    pt1 = (0,offset)
    pt2 = (width,offset)
    cv2.line(image, pt1, pt2, Color['blue'],1)     
    #return image

#_____________________Used in________________________________________________
#__Not yet
#____________________________________________________________________________
#__Filled circle draw to use in histogram sampling
#__
def MyFilledCircle(img, center, radius):
    thickness = -1
    lineType = 8
    cv2.circle(img, center, radius, Color['white'], thickness, lineType) 

#_____________________Used in________________________________________________
#__PathFinderMain():
#____________________________________________________________________________
#__Filled circle draw to use in histogram sampling
#__
def MyDraw_keypoints(vis, keypoints, radius):
    color = Color['white']
    for kp in keypoints:
            x, y = kp.pt
            cv2.circle(vis, (int(x), int(y)), radius, color,-1,8)


# Class - Text box for putting in note and accept / reject the selected area.
#_____________________________________________________________________________
class Dialog(QtGui.QDialog):
    def __init__(self):
        super(Dialog, self).__init__()


def PathFinderMain():
    config = Configuration()
    config.read()
    TrimParam = config.ConfigDictionary['TrimParam']
    BackgroundSample = config.ConfigDictionary['BackgroundSample']
    CutParam = config.ConfigDictionary['CutParam']
    CutParam['Initialised'] = False
    print '_______________________________________________Run State_______________________________________________'
    print config.RunState
    #USB camera capture 
    capture1 = cv2.VideoCapture(0)
    #capture2 = cv2.VideoCapture(1)
    #Tracker for feature detection
    tracker = PlaneTracker()


    #Configure capture screen resolution
    if ModeTest:
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1024)
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 768)
        

    else:
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
    
    gain_old = capture1.get(cv2.cv.CV_CAP_PROP_GAIN)
    while True:
        
        #check if gain cap changed
        gain_new = capture1.get(cv2.cv.CV_CAP_PROP_GAIN)
        if gain_new <> gain_old:
            print 'CV_CAP_PROP_GAIN = ',capture1.get(cv2.cv.CV_CAP_PROP_GAIN)
            gain_old = gain_new
        #print 'CV_CAP_PROP_AUTO_EXPOSURE = ',capture1.get(cv2.cv.CV_CAP_PROP_AUTO_EXPOSURE)
        #img = cv.QueryFrame(capture)
        flas,img1 = capture1.read()
        #flas,img2 = capture2.read()
        
        img_trimmed = bg.Background_Trim(img1,TrimParam)
        




        #___________________________Smooth___________________________
        #img_trimmed = bg.Background_k_mean(img_trimmed)
        #img_trimmed = cv2.GaussianBlur(img_trimmed,(3,3),0)
        #img_trimmed = cv2.bilateralFilter(img_trimmed,3, 3*2,3/2)
        #_____________________________________________________________
        
        img_trimmed_overlay = img_trimmed.copy()
        
        #___________________________Openning__________________________________
        #
        img_trimmed = bg.Background_Opening(img_trimmed, 10,2)
        #img_backprojected = bg.Background_k_mean(img_backprojected)
        bg.imshow("Open", img_trimmed,ProcessDebug)
        #_____________________________________________________________________
        
        #__________________________Draw Rect__________________________________
        #
        #Draw a black rectangular around the trimmed image to help with Canny close loop
        height, width, depth = img_trimmed.shape
        cv2.rectangle(img_trimmed, (0,0), (width,height), Color['black'],4)
        #_____________________________________________________________________
        
        #___________________________Back projection___________________________
        #Back project clear out the background using hist of sampled background image
        img_backprojected = bg.Background_remove(img_trimmed,config.Internal['Background_sample_path'])
        img_backprojected = bg.Background_k_mean(img_backprojected)
        
        #bg.imshow("BackProject", img_backprojected,ProcessDebug)
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
        bg.imshow("Canny", img_canny,ProcessDebug)
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
        bg.imshow("Dilate",img_canny,ProcessDebug)
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
                Box_hw['x'] = x
                Box_hw['y'] = y
                Box_hw['w'] = w
                Box_hw['h'] = h
                Box_hw['valid'] = True
            else:
                Box_hw['valid'] = False
                print 'Box not found'
            cv2.rectangle(img_backprojected,(x,y),(x+w,y+h),(0,255,0),2)
            bg.imshow("Show_Contour",img_backprojected,ProcessDebug)
        #____________________________________________________________
        
        #___________________________Extract object____________________________
        #only update if box h w change more than an average variation
        img_object_of = np.zeros((Box_hw['h'],Box_hw['w'],depth),np.uint8)
        if  (True)|(math.fabs(Box_hw['w'] - Box_hw['w_old']) > Box_hw['variation'])|(math.fabs(Box_hw['h'] - Box_hw['h_old']) > Box_hw['variation']):
            
            Box_hw['w_old'] = Box_hw['w']
            Box_hw['h_old'] = Box_hw['h']
            img_object_of = bg.Background_extract_obj(img_trimmed_overlay, Box_hw)
        bg.imshow ('object of interest',img_object_of,False) 

        
        #_____________________________________________________________________
        #img_trimmed_overlay_cv = cv.fromarray(img_trimmed_overlay)
        
        #___________________________Detect Feature____________________________
        #Detect feature on object of interest
        
        tracked = tracker.track(img_object_of)        
        img_object_of_feature = img_object_of.copy()
        draw_keypoints(img_object_of_feature, tracker.frame_points)
        bg.imshow ('feature',img_object_of_feature,False)
        #_____________________________________________________________________

        #___________________________Adaptive threadhold____________________________
        #Detect feature on object of interest
        
        img_object_of_adaptive = cv2.cvtColor(img_object_of,cv2.cv.CV_RGB2GRAY)
        img_object_of_adaptive_blur = cv2.medianBlur(img_object_of_adaptive,5)
 
        #ret,th1 = cv2.threshold(img_object_of_adaptive_blur,127,255,cv2.THRESH_BINARY)
        #th2 = cv2.adaptiveThreshold(img_object_of_adaptive_blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(img_object_of_adaptive_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
        th3 = cv2.medianBlur(th3,3)
        #bg.imshow ('feature_threadhold_adaptive_1',th2,True)
        bg.imshow ('feature_threadhold_adaptive_2',th3,True)
        #_____________________________________________________________________
        

        #___________________________Sample color around detected feature____________________________
        #Detect feature on object of interest
        """
        height, width, depth = img_object_of_feature.shape
        img_feature_mask = np.zeros((height,width,depth),np.uint8)
        MyDraw_keypoints(img_feature_mask, tracker.frame_points,5)
        
        
        img_of_hist = img_object_of.copy()
        img_of_hist = cv2.bitwise_and(img_of_hist,img_feature_mask)
        img_extra_feature = bg.Background_histDetect(img_object_of, img_of_hist)
        bg.imshow ('Feature Mask',img_of_hist,True)
        bg.imshow ('Feature extra',img_extra_feature,True)
        """
        
        #___________________________________________________________________________________________
        
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
        
        #______________________________________generate strip matrix__________________________

        
        if not CutParam['Initialised']:
            cut_band_pair = np.arange(CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2'] + CutParam['Strip_2'] + CutParam['Blade_3'])
            for i in range(0, len(cut_band_pair)):
                if i < CutParam['Blade_1']:
                    cut_band_pair[i] = 0
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1']):
                    cut_band_pair[i] = 1
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2']):
                    cut_band_pair[i] = 0
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2'] + CutParam['Strip_2']):
                    cut_band_pair[i] = 1
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2'] + CutParam['Strip_2'] + CutParam['Blade_3']):
                    cut_band_pair[i] = 0
                    
            cut_band_1 = np.arange(CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2'] + CutParam['Strip_2'] + CutParam['Blade_3'])
            for i in range(0, len(cut_band_1)):
                if i < CutParam['Blade_1']:
                    cut_band_1[i] = 0
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1']):
                    cut_band_1[i] = 1
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2']):
                    cut_band_1[i] = 0
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2'] + CutParam['Strip_2']):
                    cut_band_1[i] = 0
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2'] + CutParam['Strip_2'] + CutParam['Blade_3']):
                    cut_band_1[i] = 0

            cut_band_2 = np.arange(CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2'] + CutParam['Strip_2'] + CutParam['Blade_3'])
            for i in range(0, len(cut_band_2)):
                if i < CutParam['Blade_1']:
                    cut_band_2[i] = 0
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1']):
                    cut_band_2[i] = 0
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2']):
                    cut_band_2[i] = 0
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2'] + CutParam['Strip_2']):
                    cut_band_2[i] = 1
                elif i < (CutParam['Blade_1'] + CutParam['Strip_1'] + CutParam['Blade_2'] + CutParam['Strip_2'] + CutParam['Blade_3']):
                    cut_band_2[i] = 0

            CutParam['Initialised'] = True
        #print cut_band    
        #print cut_band and hist
        #now we and them small and big across the band, result matrix then summed and put in the index
        #this section addressing dual strip perfect scenario
        """
        print 'hist = '
        print hist
        print 'cut band = '
        print cut_band_pair
        print 'cut band 1 = '
        print cut_band_1
        print 'cut band 2 = '
        print cut_band_2
        """
        #for i in range(len(hist) + len(cut_band)*2):
        #    and_result = bg.Background_operator_and (cut_band, hist, i - len(cut_band))
            #print 'result = ',and_result 
        #    array_and_sum[i] = and_result.sum()
        array_and_sum = [0]*(len(hist) - len(cut_band_pair))
        for i in range(len(hist) - len(cut_band_pair)):
            and_result = bg.Background_operator_and (cut_band_pair, hist,i)
            #print 'result = ',and_result 
            array_and_sum[i] = and_result.sum()
        
        """
        array_and_sum_1 = [0]*(len(hist))
        for i in range(len(hist)):
            and_result = bg.Background_operator_and (cut_band_1, hist,i)
            #print 'result = ',and_result 
            array_and_sum_1[i] = and_result.sum()

        array_and_sum_2 = [0]*(len(hist))
        for i in range(len(hist)):
            and_result = bg.Background_operator_and (cut_band_2, hist,i)
            #print 'result = ',and_result 
            array_and_sum_2[i] = and_result.sum()

        """

        print '_______________________________________________________________________'
        print 'array_and_sum '
        print array_and_sum
        """
        print 'array_and_sum_1 '
        print array_and_sum_1
        print 'array_and_sum_2 '
        print array_and_sum_2
        """
        
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
                search_index = search_index + len(cut_band_pair)
            else:
                search_index = search_index + 1
        print "Cut List________________"
        print cut_list
        #then we illustrate the cut slide 
        for i in cut_list:
            #img_object_of_feature = Illustrate (img_object_of_feature,i)
            Illustrate (img_object_of_feature,i)
        bg.imshow ('Illustrated',img_object_of_feature,True)
        
        #_______________________________________________________________________________________________
        
        key = cv.WaitKey(2)
        #if key <> -1:
        #    print key

        if key == 27:
            break
        #Let save the configuration to file
        if key == KeyDictionary['key_ctr_s']:
            config.ConfigDictionary['TrimParam'] = TrimParam
            config.ConfigDictionary['BackgroundSample'] = BackgroundSample 
            config.write()
        #Change KeySelect function mode
        if key == KeyDictionary['key_tab']:
            config.RunStateUpdate()
        #_______________________________________ TEST FUNCTION GO HERE ________________________________
        #______________________________________________________________________________________________
        if key == KeyDictionary['key_t']:
            #img_backprojected = bg.Background_remove(img_trimmed,config.Internal['Background_sample_path'])
            #bg.imshow("BackProject", img_backprojected,ProcessDebug)
            bg.Background_Sampler(img_trimmed_overlay)
        #______________________________________________________________________________________________
        #______________________________________________________________________________________________
        #Let sample so background image
        if key == KeyDictionary['key_s']:
            img_background_sample = bg.Background_Sample(img_trimmed, BackgroundSample)
            cv2.imwrite(config.Internal['Background_sample_path'], img_background_sample) 
            cv2.imshow("Sampled", img_background_sample)
        if config.GetRunState() == 'TrimArea':
            KeyFunc_TrimArea(key,config,TrimParam)
        #Edit limit of background sample box on top
        if config.GetRunState() == 'Background_top':
            KeyFunc_Background_top(key,config,BackgroundSample)
            
            height,width,depth = img_trimmed_overlay.shape
            pt1 = (0,BackgroundSample['Top_border'])
            pt2 = (width,BackgroundSample['Top_border'])
            cv2.line(img_trimmed_overlay, pt1, pt2, Color['red'],2) 
        #Edit limit of background sample box on top
        if config.GetRunState() == 'Background_bottom':
            KeyFunc_Background_bottom(key,config,BackgroundSample)

            height,width,depth = img_trimmed_overlay.shape
            pt1 = (0,height -BackgroundSample['Bottom_border'])
            pt2 = (width,height - BackgroundSample['Bottom_border'])
            cv2.line(img_trimmed_overlay, pt1, pt2, Color['red'],2) 

        if CameraDebugScreen:
            
            cv2.imshow("usbCam_1", img_trimmed_overlay)
            
            
        

# Main program where everything start
#_____________________________________________________________________________
 
def main():
    print ("Python version = ",sys.version)
    print ("Opencv version = ",cv.__version__)
    print("Qt version = ", QT_VERSION_STR)
    print("Numpy version = ", np.version.version)
    print ("Matplotlib version = ",matplotlib.__version__)
    PathFinderMain()
 
if __name__ == "__main__":
    main()