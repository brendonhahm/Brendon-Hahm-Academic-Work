from pupil_apriltags import Detector
import cv2
import numpy as np
import os

os.chdir('C:\\Users\\brend\\Desktop\\Northeastern Class Folders\\CS 5335 Project')

image = cv2.cvtColor(cv2.imread("testing.jpg"), cv2.COLOR_BGR2GRAY)

families ='tag36h11'
camera_params= [1481.2,1481.6,952.7, 563.8]
tag_size= 0.0165 # unit: m

at_detector = Detector(families='tag36h11',
                   	nthreads=1,
                   	quad_decimate=1.0,
                   	quad_sigma=0.0,
                   	refine_edges=1,
                   	decode_sharpening=0.25,
                   	debug=0)
results = at_detector.detect(image, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

result = results[0]

#pic 1
#x:  [-70.59776291]
#y:  [-30.27220671]
#[-47.70119116]
#[-20.45419373]

#pic 2
#x:  [26.71035929]
#y:  [-36.20874761]
#[18.04754006]
#[-24.46537001]

#pic 3
#x:  [-79.28649111]
#y:  [-134.24027707]
#[-53.57195345]
#[-90.70288992]


print("x: ", result.pose_t[0] * 1000)
print("y: ", result.pose_t[1] * 1000)
print(result.pose_t[0]*1000/1.48)
print(result.pose_t[1]*1000/1.48)
((result.pose_t[0] * 1000) - 300)/100
((result.pose_t[1] * 1000)-50)/100
