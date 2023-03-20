import tinyik
import numpy as np
import xarm


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


#ik
goal = [200, 200, 200]


#x is cables perpendicular, y is cables parallel, z is cables ortho

ikarm = tinyik.Actuator(['z', [0., 0., 40.], 'y', [0., 0., 25.], 'y', [0., 0., 101.], 'y', [0., 0., 95.], 'z', [0., 0., 165.]])
tinyik.visualize(ikarm)

legs = tinyik.Actuator([[.3, .0, .0], 'z', [.3, .0, .0], 'x', [.0, -.5, .0], 'x', [.0, -.5, .0]])
#leg.angles = np.deg2rad([30, 45, -90])
tinyik.visualize(legs)
