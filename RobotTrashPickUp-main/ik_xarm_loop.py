import tinyik
import numpy as np
import xarm
import random


#ik
goal = [-2.6396, .5, -1.3]

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

arm.setPosition([[1, 0.0], [2, angles[4]], [3, angles[3]], [4, -1*angles[2]], [5, angles[1]], [6, angles[0]]], wait=True)
