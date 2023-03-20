import tinyik
import numpy as np
import xarm


#initialize servos
arm = xarm.Controller('USB')

servo1 = xarm.Servo(1)
servo2 = xarm.Servo(2)
servo3 = xarm.Servo(3)
servo4 = xarm.Servo(4)
servo5 = xarm.Servo(5, 0.)
servo6 = xarm.Servo(6)

#set position to servo object defaults
arm.setPosition([servo1, servo2, servo3, servo4, servo5, servo6])


#ik
goal = [200, 200, 200]


#x is cables perpendicular, y is cables parallel, z is cables ortho

ikarm = tinyik.Actuator([[.0, .5, .0],
                         'y', [.0, .15, .0],    #servo 6
                         'z', [.0, 1.01, .0],   #servo 5
                         'z', [.0, .95, .0],    #servo 4, negative
                         'z', [.0, .51, .0],    #servo 3
                         'y', [.0, 1.14, .0]])  #servo 2

tinyik.visualize(ikarm)
# ik
ikarm.ee = [0, 0, 2]
angles = np.round(np.rad2deg(ikarm.angles))
angles
arm.setPosition([[1, 0.0], [2, angles[4]], [3, angles[3]], [4, -1*angles[2]], [5, angles[1]], [6, angles[0]]], wait=True)
tinyik.visualize(ikarm)

# set angles to the real arm
arm.setPosition([[1, 0.0], [2, angles[4]], [3, angles[3]], [4, -1*angles[2]], [5, angles[1]], [6, angles[0]]], wait=True)

[-106.1558, -1089.3991, 0]
xrobot = -106.1558 - 300
