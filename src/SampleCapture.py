#!/usr/bin/env python
# -*- coding: UTF-8 -*-
 
 
"""
201202120259am
 
This is an example of how to gather data from various mouse events (in this
case, (x,y) coordinates), and use this data to control parameters for various
opencv functions.
 
<-*.JMJ.*->  
"""
 
 
import sys
import cv2.cv as cv
from time import gmtime, strftime
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
    
    if (event == cv.CV_EVENT_LBUTTONUP):
        Mouse_Dict_Param['MouseUpped'] = True

        
 
 
# Generate the viewing window and designate the video source.
#_____________________________________________________________________________
 
def create_window(name,width,height):
    cv.NamedWindow(name, 1)
    stream = cv.CreateCameraCapture(0)
    cv.SetCaptureProperty(stream, cv.CV_CAP_PROP_FRAME_HEIGHT, height)
    cv.SetCaptureProperty(stream,cv. CV_CAP_PROP_FRAME_WIDTH, width)
    cv.NamedWindow(name + "_Status", 1)
    return stream
 
 
# Stream the video, already dangit!
#_____________________________________________________________________________
 
def get_webcam_stream(width,height):
 
 
    stream = create_window('face_camera',width,height)
    
    
    output = cv.QueryFrame(stream)
    cv.SetMouseCallback("face_camera", onMove, Mouse_Dict_Param)
    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.3, 0.3, 0, 1)
    
    # Live feed loop.
    while True:
        output = cv.QueryFrame(stream)
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
        print ("k =",k)
        if k == 102: #f
            break
        if k == 114: #r for reset drawing
            Mouse_Dict_Param['Draw_Rect']['DrawState'] = False
            Mouse_Dict_Param['Draw_Rect']['DrawCompleted'] = False
        if k == 115: #s for save image of drawing
            Mouse_Dict_Param['Draw_Rect']['DrawState'] = False
            Mouse_Dict_Param['Draw_Rect']['DrawCompleted'] = False
            Time_Str = strftime("%Y-%m-%d_%H%M%S", gmtime())
            
        
         
# ::: :
#_____________________________________________________________________________
 
def main():
 
    get_webcam_stream(640,480)
 
if __name__ == "__main__":
    main()