from pupil_apriltags import Detector
import cv2
import numpy as np
import os
import tinyik
import xarm
import random


#initialize servos
arm = xarm.Controller('USB')

servo1 = xarm.Servo(1)
servo2 = xarm.Servo(2)
servo3 = xarm.Servo(3)
servo4 = xarm.Servo(4)
servo5 = xarm.Servo(5)
servo6 = xarm.Servo(6)

#set position to servo object defaults
arm.setPosition([servo1, servo2, servo3, servo4, servo5, servo6])


os.chdir('C:\\Users\\brend\\Desktop\\Northeastern Class Folders\\CS 5335 Project')

image = cv2.cvtColor(cv2.imread("merge.jpg"), cv2.COLOR_BGR2GRAY)

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
x = ((result.pose_t[0] * 1000) - 300)/100
y = ((result.pose_t[1] * 1000) - 60)/100






#ik
goal = [x, .5, y]

ikarm = tinyik.Actuator([[.0, .5, .0],
                         'y', [.0, .15, .0],    #servo 6
                         'z', [.0, 1.01, .0],   #servo 5
                         'z', [.0, .95, .0],    #servo 4, negative
                         'z', [.0, .51, .0],    #servo 3
                         'y', [.0, 1.14, .0]])  #servo 2
for i in range(5000):
    exit = False
    ikarm.ee = goal
    angles = np.round(np.rad2deg(ikarm.angles))

    new_angles = []
    for ang in angles:
        if ang >= 0:
            new_angles.append(ang%360)
        else:
            ang_flip = -1*ang
            new_angles.append((ang_flip%360)* -1)

    for ang in new_angles:
        if ang > 125 or ang <-125:
            exit = True
    if exit == True:
        ikarm.ee = [random.randint(0, 4), random.randint(0, 4), random.randint(0, 4)]
        continue
    else:
        break

angles = new_angles





#initialize servos
#arm = xarm.Controller('USB')

servo1 = xarm.Servo(1, 1000)
servo2 = xarm.Servo(2, 0.0)
servo3 = xarm.Servo(3, 0.)
servo4 = xarm.Servo(4, 0.)
servo5 = xarm.Servo(5, 0.)
servo6 = xarm.Servo(6, 0.)

#set position to servo object defaults
arm.setPosition([servo1, servo2, servo3, servo4, servo5, servo6])

arm.setPosition([[1, 0], [2, angles[4]], [3, angles[3]], [4, -1*angles[2]], [5, angles[1]], [6, angles[0]]], wait=True)
