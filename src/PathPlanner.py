import sys
import cv2.cv as cv
import cv2
import numpy as np

from time import gmtime, strftime

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import QT_VERSION_STR

# Background module which handle sampling background histogram / filtering
import Background as bg
# Class - Text box for putting in note and accept / reject the selected area.
#_____________________________________________________________________________
class Dialog(QtGui.QDialog):
    def __init__(self):
        super(Dialog, self).__init__()


def PathFinderMain():
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
    
    
        if CameraDebugScreen:
            cv2.imshow("usbCam_1", img1)


        if cv.WaitKey(2) == 27:
            break
        


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