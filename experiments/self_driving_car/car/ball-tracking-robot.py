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

# print (args['video'])

cap = cv2.VideoCapture(int(args['video']))
# cap = cv2.VideoCapture(2)

def get_theta(dx, dy):
    #if frame is being mirror
    dx = -dx
    return math.atan(dx/dy) * 180 / 3.14

while True:
    ret, frame = cap.read()
    #flip miror
    frame = cv2.flip(frame, 1)

    screen_width = int(cap.get(3))   
    screen_height = int(cap.get(4)) 
    x_screen_center = int(screen_width/2)
    y_screen_center = int(screen_height/2)

    cv2.line(frame,(x_screen_center,0),(x_screen_center,screen_height),(255,0,0),5)

    blur = cv2.GaussianBlur(frame, (11, 11), 0)

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
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, ball_center, 5, (0, 0, 255), -1)
 
    distance = 0
    if(x_ball_center != -1):
        distance = x_ball_center - x_screen_center 
    
        dx = distance
        dy = (screen_height - y_ball_center) / 4
        # if (dy > screen_width/2):
        #     dy = screen_width/2

        theta = get_theta(dx, dy)
        speed = 20
    else:
        speed = 0
    # if theta > 0:
    #     print('right')
    # else:
    #     print('left')
    if old_speed != speed:
        old_speed = servo_controller.set_speed(speed)

    if old_theta != theta:
        old_theta = servo_controller.set_steer(theta)
    # if distance > 50:
    #     print ('right')
    #     theta = servo_controller.set_steer(30)

    # elif distance < -50:
    #     print ('left')
    #     theta = servo_controller.set_steer(-30)
        
    # print (distance)

    cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()