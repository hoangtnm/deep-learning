import numpy as np
import cv2
import imutils
import math
from gpio_controller.ServoController import ServoController
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', default=1)
args = vars(ap.parse_args())


servo_controller = ServoController()
servo_controller.init()
theta = 0
speed = 0
old_theta = 0
old_speed = 0

SCREEN_WIDTH=240
SCREEN_HEIGHT=320

CAMERA_MIDDLE = 2
CAMERA_LEFT = 1
CAMERA_RIGHT = 3
# print (args['video'])
#3 - right
#1 - middle
#2 - left
cap_left = cv2.VideoCapture(CAMERA_LEFT) 
cap_left.set(3, SCREEN_WIDTH)
cap_left.set(4, SCREEN_HEIGHT)

cap_middle = cv2.VideoCapture(CAMERA_MIDDLE) 
cap_middle.set(3, SCREEN_WIDTH)
cap_middle.set(4, SCREEN_HEIGHT)

cap_right = cv2.VideoCapture(CAMERA_RIGHT) 
cap_right.set(3, SCREEN_WIDTH)
cap_right.set(4, SCREEN_HEIGHT)

screen_width = int(cap_middle.get(3))   
screen_height = int(cap_middle.get(4)) 
x_screen_center = int(screen_width/2)
y_screen_center = int(screen_height/2)


# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture(2)

def get_theta(dx, dy):
    #if frame is being mirror
    dx = -dx
    return math.atan(dx/dy) * 180 / 3.14

def get_theta_from_camera(image, camera_pos):
    blur = cv2.GaussianBlur(image, (11, 11), 0)

    hsv = cv2.cvtColor( blur, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 100, 150])
    upper = np.array([25, 255, 255])

    mask = cv2.inRange( hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    ball_cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  	
    ball_cnts = imutils.grab_contours(ball_cnts)
    ball_center = None
    x_ball_center = -1
    y_ball_center = -1

    if len(ball_cnts) > 0:
        c = max(ball_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        x_ball_center = int(M["m10"] / M["m00"])
        y_ball_center = int(M["m01"] / M["m00"])
        ball_center = (x_ball_center,y_ball_center)
 
		# only proceed if the radius meets a minimum size
        if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(image, ball_center, 5, (0, 0, 255), -1)
    
    theta = None
    if x_ball_center == -1:
        return image, theta
    
    if camera_pos == CAMERA_MIDDLE:
        distance = x_ball_center - x_screen_center 
    
        dx = distance
        dy = (screen_height - y_ball_center)
        # if (dy > screen_width/2):
        #     dy = screen_width/2

        theta = get_theta(dx, dy)
        
    elif camera_pos == CAMERA_LEFT:
        dx = x_ball_center - screen_width * 3/2
        dy = (screen_height - y_ball_center)
        theta = get_theta(dx, dy)
    
    elif camera_pos == CAMERA_RIGHT:
        dx = x_ball_center + screen_width * 1/2
        dy = (screen_height - y_ball_center)
        theta = get_theta(dx, dy)

    return image, theta


while True:
    ret, frame_middle = cap_middle.read()
    ret, frame_left = cap_left.read()
    ret, frame_right = cap_right.read()

      #flip miror
    frame_middle = cv2.flip(frame_middle, 1)
    frame_left = cv2.flip(frame_left, 1)
    frame_right = cv2.flip(frame_right, 1)

    

    cv2.line(frame_middle,(x_screen_center,0),(x_screen_center,screen_height),(255,0,0),5)

    '''
    blur = cv2.GaussianBlur(frame_middle, (11, 11), 0)

    hsv = cv2.cvtColor( blur, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 100, 150])
    upper = np.array([25, 255, 255])

    mask = cv2.inRange( hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    ball_cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  	
    ball_cnts = imutils.grab_contours(ball_cnts)
    ball_center = None
    x_ball_center = -1
    y_ball_center = -1

    if len(ball_cnts) > 0:
        c = max(ball_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        x_ball_center = int(M["m10"] / M["m00"])
        y_ball_center = int(M["m01"] / M["m00"])
        ball_center = (x_ball_center,y_ball_center)
 
		# only proceed if the radius meets a minimum size
        if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
            cv2.circle(frame_middle, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame_middle, ball_center, 5, (0, 0, 255), -1)
    
    '''
    theta = None

    frame_middle, theta_middle = get_theta_from_camera(frame_middle, CAMERA_MIDDLE)
    frame_left, theta_left = get_theta_from_camera(frame_left, CAMERA_LEFT)
    frame_right, theta_right = get_theta_from_camera(frame_right, CAMERA_RIGHT)

    if theta_middle is not None:
        theta = theta_middle
    elif theta_left is not None:
        theta = theta_left
    elif theta_right is not None:
        theta = theta_right

    distance = 0
    if theta is not None:
        speed = 20
    else:
        speed = 0

    if old_speed != speed:
        old_speed = servo_controller.set_speed(speed)

    if (theta is not None) and (old_theta != theta):
        old_theta = servo_controller.set_steer(theta)

    old_theta = theta

    cv2.imshow('middle', frame_middle)
    cv2.imshow('left', frame_left)
    cv2.imshow('right', frame_right)

    # cv2.imshow('mask', mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()