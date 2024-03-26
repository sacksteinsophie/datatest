# Usage
# python scripts/multi_tracking.py -v 'resources/group_of_people_05.mp4' -t 'csrt'

import cv2
import sys
import argparse
from random import randint
import datetime
import mxcamera
#########################################################################################################
#################################### Select Tracking Algorithm ##########################################
#########################################################################################################
import time

def current_milli_time():
    return round(time.time() * 1000)
start_time = current_milli_time()
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
mxcameraclass = mxcamera.mxclass(0,0)
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,  default='udp://127.0.0.1:5000', help="URl or Path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
args = vars(ap.parse_args())

def create_tracker_by_name(tracker_type):
    if tracker_type == tracker_types[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == tracker_types[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == tracker_types[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == tracker_types[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == tracker_types[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == tracker_types[5]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == tracker_types[6]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('[ERROR] Invalid selection! Available tracker: ')
        for t in tracker_types:
            print(t.lower())

    return tracker

print('[INFO] selected tracker: ' + str(args["tracker"].upper()))


#########################################################################################################
############################################# Load Video ################################################
#########################################################################################################

video = cv2.VideoCapture(args["video"],cv2.CAP_FFMPEG)
# load video
if not video.isOpened():
    print('[ERROR] video file not loaded')
    sys.exit()
ok, frame = video.read()
if not ok:
    print('[ERROR] no frame captured')
    sys.exit()

print('[INFO] video loaded and frame capture started')

# set recording parameter
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
video_codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
prefix = 'recording/'+datetime.datetime.now().strftime("%y%m%d_%H%M%S")
basename = "object_track.mp4"
video_output = cv2.VideoWriter("_".join([prefix, basename]), video_codec, fps, (frame_width, frame_height))
height = frame_height
width = frame_width
# The variables correspond to the middle of the screen, and will be compared to the midFace values
midScreenY = (height/2)
midScreenX = (width/2)
midScreenWindow = 50 # This is the acceptable 'error' for the center of the screen
stepSize=1

#########################################################################################################
###################################### Select Objects to track ##########################################
#########################################################################################################

# Define list for bounding boxes
# and tracking rectangle colour
bboxes = []
colours = []
name_strings =[]
f = open("name.txt", 'r')
label_num = 0 
while True:
    # Open selector and wait for selected ROIs
    bbox = cv2.selectROI('MultiTracker', frame)
    print('[INFO] select ROI')
    print('[INFO] press SPACE or ENTER to confirm selection')
    print('[INFO] press q to exit selection or any other key to continue')
    
    # Add ROIs to list of bounding boxes
    font = cv2.FONT_HERSHEY_SIMPLEX
    nameslist = [i for i in f.read().split("\n")]
    
    print(nameslist)
    
    try:
        name_str = nameslist[label_num]
    except: 
        name_str = "unlabelled"
    print(bbox)
    if int(bbox[0]+bbox[1])>1:
        label_num +=1
        #except: 
        #name_str = "unlabelled"
        name_strings.append(name_str)
        bboxes.append(bbox)
        # Create random colour for each box
        colours.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    # Wait until user presses q to quit selection
    k = cv2.waitKey(0) & 0xff
    if k == ord('q'):
        break


#########################################################################################################
####################################### Initialize ROI Tracking #########################################
#########################################################################################################
print("##########################")
print("name_strings")
print(name_strings)
print("##########################")
if len(bboxes) >0: 
    multi_tracker = cv2.legacy.MultiTracker_create()

    for bbox in bboxes:
        multi_tracker.add(create_tracker_by_name(args["tracker"].upper()), frame, bbox)
    frame_num =0
    start_time = current_milli_time()

    while video.isOpened():
        # get frames from video
        ok, frame = video.read()
        if not ok:
            print('[INFO] end of video file reached')
            break

        # get new bounding box coordinates
        # for each frame from tracker
        ok, boxes = multi_tracker.update(frame)
        if not ok:
            cv2.putText(frame, 'Track Loss', (10, 50), cv2.QT_FONT_NORMAL, 1, (0, 0, 255))
        # use coordinates to draw rectangle
        #print(boxes)
        for i, new_box in enumerate(boxes):
            (x, y, w, h) = [int(v) for v in new_box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), colours[i], 3)
            TopOfText = (x-h,y)
            fontScale= 1
            fontColor = (0,0,0)
            thickness = 1
            lineType = 2
            
            cv2.putText(frame,name_strings[i], 
            TopOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
            annotations = {"class": name_strings[i], 
            "subclass": "RHIB", 
            "size_meters": 7, 
            "environment": "open-sea",
            "frame": frame_num, 
            "timestamp": current_milli_time() - start_time, 

            "annotations": {"bounding_box": [int(v) for v in new_box]}
            }
            print(annotations)
            
             #so we manipulate these values to find the midpoint of the rectangle
        chosen_box = boxes[0] 
        (x, y, w, h) = [int(v) for v in chosen_box]
        frame_num +=1

        midFaceY = y + (height/2)
        midFaceX = x + (width/2)

        
        #print(mxcameraclass.servoTiltPosition, mxcameraclass.servoPanPosition)
        cv2.putText(frame, str(args["tracker"].upper()), (10, 30), cv2.QT_FONT_NORMAL, 1, (255, 255, 255))
        # record object track
        video_output.write(frame)
        cv2.imshow("MultiTracker", frame)
        # press 'q' to break loop and close window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()