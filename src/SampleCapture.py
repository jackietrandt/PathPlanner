#!/usr/bin/env pythongoogle
# -*- coding: UTF-8 -*-
 

 
import sys
import cv2.cv as cv
from time import gmtime, strftime

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import QT_VERSION_STR

# Class - Text box for putting in note and accept / reject the selected area.
#_____________________________________________________________________________
class Dialog(QtGui.QDialog):
    def __init__(self):
        super(Dialog, self).__init__()

        self.bigEditor = QtGui.QTextEdit()
        self.bigEditor.setPlainText("Defect : ")

        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        mainLayout = QtGui.QVBoxLayout()
        mainLayout.addWidget(self.bigEditor)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

        self.setWindowTitle("Image Note")
        
        self.AcceptResult = False
    def accept(self): # subclass QDialog and define the accept there
        self.AcceptResult = True
        print "Accept"
        #print self.bigEditor.toPlainText() # print out the note when accepted
        super(Dialog, self).accept() # call the accept method of QDialog.super is needed since we just override the accept method 
    def reject(self):
        self.AcceptResult = False
        print "Reject"
        super(Dialog, self).reject() # call the accept method of QDialog.super is needed since we just override the accept method 
        
# function to draw the rectangle, added flag -1 to fill rectangle. If you don't want to fill, just delete it.
#_____________________________________________________________________________
box=[0,0,0,0]

def draw_box(img,box):
    cv.Rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)

# Define mouse events - Data structure
#_____________________________________________________________________________
Mouse_Dict_Param = {'Mouse_x':0,
                    'Mouse_y':0,
                    'Action_ID':0,
                    'Draw_Rect':{},
                    'MouseUpped':True
                    }
Mouse_Dict_Param['Draw_Rect'] = {'Rect':{},
                                 'DrawState': False,
                                 'DrawCompleted': False
                                 }
Mouse_Dict_Param['Draw_Rect']['Rect'] = {'x1':0,
                                         'y1':0,
                                         'x2':0,
                                         'y2':0}
# Define mouse events.
#_____________________________________________________________________________
 
def onMove(event, x, y, flags, Mouse_Dict_Param):
    
    Mouse_Dict_Param['Mouse_x'] = x
    Mouse_Dict_Param['Mouse_y'] = y
    
    if (event == cv.CV_EVENT_LBUTTONDOWN) & Mouse_Dict_Param['MouseUpped'] & ~Mouse_Dict_Param['Draw_Rect']['DrawCompleted']:
        Mouse_Dict_Param['MouseUpped'] = False
        MatchFound = False
        if Mouse_Dict_Param['Draw_Rect']['DrawState'] == False:
            Mouse_Dict_Param['Draw_Rect']['Rect']['x1'] = x
            Mouse_Dict_Param['Draw_Rect']['Rect']['y1'] = y
            Mouse_Dict_Param['Draw_Rect']['DrawState'] = True
            MatchFound = True
        if (Mouse_Dict_Param['Draw_Rect']['DrawState'] == True) & (MatchFound == False):
            Mouse_Dict_Param['Draw_Rect']['Rect']['x2'] = x
            Mouse_Dict_Param['Draw_Rect']['Rect']['y2'] = y
            Mouse_Dict_Param['Draw_Rect']['DrawState'] = False
            Mouse_Dict_Param['Draw_Rect']['DrawCompleted'] = True
            #swap rect coordinate if top is lower than bottom
            x1 = Mouse_Dict_Param['Draw_Rect']['Rect']['x1']
            x2 = Mouse_Dict_Param['Draw_Rect']['Rect']['x2']
            y1 = Mouse_Dict_Param['Draw_Rect']['Rect']['y1']
            y2 = Mouse_Dict_Param['Draw_Rect']['Rect']['y2']
            if x1 > x2:
                Mouse_Dict_Param['Draw_Rect']['Rect']['x1'] = x2
                Mouse_Dict_Param['Draw_Rect']['Rect']['x2'] = x1
            if y1 > y2:
                Mouse_Dict_Param['Draw_Rect']['Rect']['y1'] = y2
                Mouse_Dict_Param['Draw_Rect']['Rect']['y2'] = y1
                
    
    if (event == cv.CV_EVENT_LBUTTONUP):
        Mouse_Dict_Param['MouseUpped'] = True
 
# Generate the viewing window and designate the video source.
#_____________________________________________________________________________
 
def create_window(name,width,height):
    cv.NamedWindow(name, 1)
    stream = cv.CreateCameraCapture(0)
    cv.SetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    cv.SetCaptureProperty(stream,cv. CV_CAP_PROP_FRAME_WIDTH, width)
    return stream
 
 
# Stream the video, already dangit!
#_____________________________________________________________________________
 
def get_webcam_stream(width,height):
 
    
    stream = create_window('face_camera',width,height)
    
    output = cv.QueryFrame(stream)
    output_clone = cv.CreateImage((output.width,output.height),8,3)
    #set mount call back Event
    cv.SetMouseCallback("face_camera", onMove, Mouse_Dict_Param)
    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.3, 0.3, 0, 1)
    #init edit text box 
    app = QtGui.QApplication(sys.argv)
    dialog = Dialog()       
    # Live feed loop.
    while True:
        output = cv.QueryFrame(stream)
        cv.Copy(output, output_clone)
        stream_status = cv.CreateImage((200,height),8,3)
        cv.Set(stream_status, (0,0,0))
        #General status report
        cv.PutText(stream_status, ("x:y =  %s:%s" % (Mouse_Dict_Param['Mouse_x'],Mouse_Dict_Param['Mouse_y'])), (5,10), font, cv.RGB(255, 0, 0))
        #cv.ShowImage('face_camera', output)
        if Mouse_Dict_Param['Draw_Rect']['DrawState'] == True:
            draw_box(output,
                     (Mouse_Dict_Param['Draw_Rect']['Rect']['x1'],
                      Mouse_Dict_Param['Draw_Rect']['Rect']['y1'],
                      Mouse_Dict_Param['Mouse_x'],
                      Mouse_Dict_Param['Mouse_y']))
        if Mouse_Dict_Param['Draw_Rect']['DrawCompleted'] == True:
            draw_box(output,
                     (Mouse_Dict_Param['Draw_Rect']['Rect']['x1'],
                      Mouse_Dict_Param['Draw_Rect']['Rect']['y1'],
                      Mouse_Dict_Param['Draw_Rect']['Rect']['x2'],
                      Mouse_Dict_Param['Draw_Rect']['Rect']['y2']))
            
        cv.ShowImage("face_camera", output)
        cv.ShowImage("face_camera_Status", stream_status)
        
        #print ("Mouse Upped =",Mouse_Dict_Param['MouseUpped'])
        #print ("Mouse State Drawing =",Mouse_Dict_Param['Draw_Rect']['DrawState'])
        #print ("Mouse Draw complete =",Mouse_Dict_Param['Draw_Rect']['DrawCompleted'])
 
        k = cv.WaitKey(1)
        
        if k == 102: #f
            break
        if k == 114: #r for reset drawing
            Mouse_Dict_Param['Draw_Rect']['DrawState'] = False
            Mouse_Dict_Param['Draw_Rect']['DrawCompleted'] = False
        if k == 115: #s for save image of drawing
            #Initialise Note dialog box 
            #then run it
            #sys.exit(dialog.exec_())
            dialog.bigEditor.setPlainText("Defect : ")
            dialog.exec_() #this dialog will return with accept or reject
            if dialog.AcceptResult:
                plaintext = dialog.bigEditor.toPlainText() #read plain text then to write to file
                print plaintext #print out the note onto console
            
                ROI_x = Mouse_Dict_Param['Draw_Rect']['Rect']['x1']
                ROI_y = Mouse_Dict_Param['Draw_Rect']['Rect']['y1']
                ROI_w = Mouse_Dict_Param['Draw_Rect']['Rect']['x2'] - ROI_x
                ROI_h = Mouse_Dict_Param['Draw_Rect']['Rect']['y2'] - ROI_y
            
                #used clone image to save without drawed rect
                cv.SetImageROI(output_clone, (ROI_x,ROI_y,ROI_w,ROI_h)) 
                Time_Str_jpg = strftime("data\%Y-%m-%d_%H%M%S.jpg", gmtime())
                Time_Str_txt = strftime("data\%Y-%m-%d_%H%M%S.txt", gmtime())
                #Save image to jpg file
                cv.SaveImage(Time_Str_jpg,output_clone)
                #Save defect description file
                f = open(Time_Str_txt, 'w')
                f.write(plaintext)
                f.close()
                cv.ResetImageROI(output_clone)
            #Then reset the draw state to not display the marking box anymore
            Mouse_Dict_Param['Draw_Rect']['DrawState'] = False
            Mouse_Dict_Param['Draw_Rect']['DrawCompleted'] = False
            
            
# Main program where everything start
#_____________________________________________________________________________
 
def main():
    print ("Python version = ",sys.version)
    print ("Opencv version = ",cv.__version__)
    print("Qt version = ", QT_VERSION_STR)
    get_webcam_stream(640,480)
 
if __name__ == "__main__":
    main()