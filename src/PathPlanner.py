import sys
import cv2.cv as cv
import cv2
import numpy as np
import math
from time import gmtime, strftime

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import QT_VERSION_STR


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
         'black':(0,0,0)
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


#Configuration class - to hold initial startup configuration when project first load / run
class Configuration(object):
    def __init__(self):
        #Define what in the config dictionary
        self.ConfigDictionary = {'TrimParam':{},
                                 'BackgroundSample':{}}
        #Define the structure of each config block
        self.ConfigDictionary['TrimParam'] = TrimParam
        self.ConfigDictionary['BackgroundSample'] = BackgroundSample
        
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
            raise 
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
    print '_______________________________________________Run State_______________________________________________'
    print config.RunState
    #USB camera capture 
    capture1 = cv2.VideoCapture(0)
    capture2 = cv2.VideoCapture(1)
    

    #________Program configuration option______
    ModeTest = True # set capture frame resolution 768 x 1024 ___or 1080 x 1920
    CameraDebugScreen = True #show raw capture from camera 1 and 2 for alignment and checking ___or disable at run mode
    ProcessDebug = True #show processed image in the middle
    #Configure capture screen resolution
    if ModeTest:
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1024)
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 768)
        

    else:
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
    #disable autofocus and auto exposer gain white balance stuff
    
    #Test run background sample which sample background color histogram
    #flag, img_background = capture1.read()
    #bg.Background_Sampler(img_background)
    #cap = cv.CaptureFromCAM(-1)
    #cv.SetCaptureProperty(cap, cv.CV_CAP_PROP_AUTO_EXPOSURE, 0);
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
        flas,img2 = capture2.read()
        
        img_trimmed = bg.Background_Trim(img1,TrimParam)
        

        #Draw a black rectangular around the trimmed image to help with Canny close loop
        height, width, depth = img_trimmed.shape
        cv2.rectangle(img_trimmed, (0,0), (width,height), Color['black'],4)
        #___________________________Smooth___________________________
        #img_trimmed = bg.Background_k_mean(img_trimmed)
        #img_trimmed = cv2.GaussianBlur(img_trimmed,(3,3),0)
        #img_trimmed = cv2.bilateralFilter(img_trimmed,3, 3*2,3/2)
        #_____________________________________________________________
        
        img_trimmed_overlay = img_trimmed.copy()
        
        #___________________________Back projection___________________________
        #Back project clear out the background using hist of sampled background image
        img_backprojected = bg.Background_remove(img_trimmed,config.Internal['Background_sample_path'])
        #img_backprojected = bg.Background_k_mean(img_backprojected)
        bg.imshow("BackProject", img_backprojected,ProcessDebug)
        #_____________________________________________________________________

        #___________________________morphologyEx_______________________________
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        res = cv2.morphologyEx(img_backprojected,cv2.MORPH_OPEN,kernel)
        bg.imshow("morphologyEx", res,ProcessDebug)
        #______________________________________________________________________
        
        #___________________________Smooth___________________________
        #img_trimmed = bg.Background_k_mean(img_trimmed)
        res = cv2.GaussianBlur(res,(3,3),0)
        #res = cv2.bilateralFilter(res,3, 3*2,3/2)
        #_____________________________________________________________        
        
        #___________________________Canny___________________________
        #img_trimmed = bg.Background_k_mean(img_trimmed)
        img_canny = cv2.Canny(res, 80, 200)
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
        kernel = np.ones((11,11),'int')
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
        if  (math.fabs(Box_hw['w'] - Box_hw['w_old']) > Box_hw['variation'])|(math.fabs(Box_hw['h'] - Box_hw['h_old']) > Box_hw['variation']):
            
            Box_hw['w_old'] = Box_hw['w']
            Box_hw['h_old'] = Box_hw['h']
            img_object_of = bg.Background_extract_obj(img_trimmed, Box_hw)
        

        bg.imshow ('object of interest',img_object_of,ProcessDebug)
        #_____________________________________________________________________
        #img_trimmed_overlay_cv = cv.fromarray(img_trimmed_overlay)
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
            img_backprojected = bg.Background_remove(img_trimmed,config.Internal['Background_sample_path'])
            bg.imshow("BackProject", img_backprojected,ProcessDebug)
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
    PathFinderMain()
 
if __name__ == "__main__":
    main()