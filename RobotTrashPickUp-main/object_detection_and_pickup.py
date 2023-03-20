# -*- coding: utf-8 -*-

import numpy as np
import time
import os
import cv2
import xarm
import tinyik
from pupil_apriltags import Detector



# ========== helper function ==========
# inverse kinematics
# input is the target end effector's position
# output is the joint angles
def ik(pos):
    # loop until find a reachable configuration
    for i in range(max_ite):
        arm.ee = pos
        angles = np.round(np.rad2deg(arm.angles))
        checked_angles = []

        # check if angles are out of range, reachable angle range [-125°, 125°]
        inRange = True
        for angle in angles:
            angle = 1.0 * angle % 360
            if angle > 180:
                angle = angle - 360
            checked_angles.append(angle)

            if angle < -125 or angle > 125:
                inRange = False
                break

        if inRange:
            return checked_angles

        if i == max_ite - 1:
            print("no possible configuration found")
    return []




# move the arm to a target position
# input is joint angles and servo1 setting
# the funtion will move the arm to that configuration
def move(angles, servo1):
    # servo1 controls grip width
    xarm.setPosition([[1, servo1], [2, angles[4]], [3, angles[3]], [4, -1*angles[2]], [5, angles[1]], [6, angles[0]]], wait=True)


# ========== arm simulation using tinyik ==========
# we measure link lengths manually and check with online product description
# we don't simulate the servo that controls the grip switch (servo 1)
# the joints order is different from the arm's, the comments below show the correspondence
# the unit of lengths is decimeters
# the servo4's positive x direction is different from others, so we need to times joint4's angle with -1
arm = tinyik.Actuator([[.0, .5, .0],
                         'y', [.0, .15, .0],    #servo 6
                         'z', [.0, 1.01, .0],   #servo 5
                         'z', [.0, .95, .0],    #servo 4, negative
                         'z', [.0, .51, .0],    #servo 3
                         'y', [.0, 1.14, .0]])  #servo 2
# uncomment below line to see arm simulation result, close the visulization window before running other codes
# tinyik.visualize(arm)

# connect to our xarm
xarm = xarm.Controller('USB')
print('Battery voltage in volts:', xarm.getBatteryVoltage())

# different settings for arm grip
grip_close = 1000
grip_open = 0
grip_grab = 475
default_pos = xarm.setPosition([[1, grip_grab], [2, 500], [3, 500], [4, 500], [5, 500], [6, 500]], wait=True)
default_pos
max_ite = 200

os.chdir('C:\\Users\\brend\\Desktop\\Northeastern Class Folders\\CS 5335 Project')
object_image_name = 'chicken.jpg'

# ========== get object's coordinates in arm base frame ==========
# take a photo of the object using webcam
# define a video capture object
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# take pic
ret, frame = vid.read()
# store pic
cv2.imwrite(object_image_name, frame)

# release the cap object
vid.release()
# destroy all the windows
cv2.destroyAllWindows()

# get the coordinates of the object in webcam's frame with AprilTag
image = cv2.cvtColor(cv2.imread(object_image_name), cv2.COLOR_BGR2GRAY)

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

# omit the height
pos_in_webcam = [result.pose_t[0]*1000, result.pose_t[1]*1000, result.pose_t[2]*1000]

# change the coordinates frame from webcam to arm base

#regular axes
pos_in_base = [-1*(pos_in_webcam[1]+300)/100, 0, (pos_in_webcam[0])/100] # unit: dm

print('webcam pos in mm: ', pos_in_webcam[0], pos_in_webcam[1], pos_in_webcam[2])
print('base pos: ', pos_in_base)

xarm.setPosition([[1, grip_grab], [2, 500], [3, 500], [4, 500], [5, 500], [6, 500]], wait=True)

# compute joint angles using inverse kinematics
# set the target position a little bit higher the table
object_up_pos = [pos_in_base[0], 1, pos_in_base[2]]
object_up_angles = ik(object_up_pos)
print(object_up_angles)
if len(object_up_angles) == 0:
    print("no possible configuration for current object position")

# move the arm grip to the object
move(object_up_angles, grip_close)

# open the grip, grip servo is servo1
xarm.setPosition(1, grip_open, wait=True)

# go down a bit
object_pos = [pos_in_base[0], 0.3, pos_in_base[2]]
object_angles = ik(object_pos)
print(object_angles)
if len(object_angles) == 0:
    print("no possible configuration for current object position")
move(object_angles, grip_open)

# close the grip and raise the arm
xarm.setPosition(1, grip_grab, wait=True)
time.sleep(1)
xarm.setPosition([[1, grip_grab], [2, 500], [3, 500], [4, 500], [5, 500], [6, 500]], wait=True)
time.sleep(1)

# move to the designated area
# a bit higher than the table
area_pos = [-2.6, 0.5, 1]  # predefined
area_angles = ik(area_pos)
print(area_angles)
if len(area_angles) == 0:
    print("no possible configuration for current drop area position")
move(area_angles, grip_grab)

# open the grip tp drop
xarm.setPosition(1, grip_open, wait=True)
xarm.setPosition([[1, grip_grab], [2, 500], [3, 500], [4, 500], [5, 500], [6, 500]], wait=True)
