import sys
import cv2.cv as cv
import cv2
import numpy as np

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

Box = {'x1':0,
       'y1':0,
       'x2':0,
       'y2':0
      }

Color = {'red':(0,0,255)
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
BackgroundSample['Box_top'] = Box
BackgroundSample['Box_bottom'] = Box


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
    #Configure capture screen resolution
    if ModeTest:
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1024)
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 768)


    else:
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920)
        capture1.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080)
        
    #Test run background sample which sample background color histogram
    flag, img_background = capture1.read()
    bg.Background_Sampler(img_background)
    
    while True:
    
        #img = cv.QueryFrame(capture)
        flas,img1 = capture1.read()
        flas,img2 = capture2.read()
        
        img_trimmed = bg.Background_Trim(img1,TrimParam)
        img_trimmed_overlay = img_trimmed.copy()
        
        img_backprojected = bg.Background_remove(img_trimmed,config.Internal['Background_sample_path'])
        cv2.imshow("BackProject", img_backprojected)
        #img_trimmed_overlay_cv = cv.fromarray(img_trimmed_overlay)
        key = cv.WaitKey(2)
        if key <> -1:
            print key

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
            cv2.imshow("BackProject", img_backprojected)
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