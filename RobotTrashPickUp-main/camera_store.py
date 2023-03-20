import os
os.chdir('C:\\Users\\brend\\Desktop\\Northeastern Class Folders\\CS 5335 Project')

# import the opencv library
import cv2


# define a video capture object
vid = cv2.VideoCapture(1)

#take pic
ret, frame = vid.read()


cv2.imwrite('frame.jpg', frame)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
