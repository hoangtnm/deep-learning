import os
import sys

curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(curr_dir)

import cv2
import argparse
from gpio_controller.ServoController import ServoController

def run(speed_add=1, theta_add=20):
    servo_controller = ServoController()
    servo_controller.init()

    speed = 0
    theta = 0
    oldSpeed = 0
    oldTheta = 0
    # speed_add = 10
    # theta_add = 20

    cap = cv2.VideoCapture(1)
    while True:
        _, frame = cap.read()

        cv2.putText(frame, f'Speed: {speed}, Theta: {theta}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('data', frame)

        command  = cv2.waitKey(1) & 0xFF
        

        if command == ord('q'):
            servo_controller.set_speed(0)
            servo_controller.set_steer(0)
            break
        
        if (chr(command) == 'w' or command == 82):
            speed += speed_add
                    
        elif (chr(command) == 's' or command == 84):
            speed -= speed_add

        elif (chr(command) == 'd' or command == 83):
            theta += theta_add

        elif (chr(command) == 'a' or command == 81):
            theta -= theta_add

        elif (command == ord('\n') or command == ord('\r')):
            theta = 0
            speed = 0
                
        if (oldSpeed != speed) or (oldTheta != theta):
            oldSpeed = speed
            oldTheta = theta

            speed = servo_controller.set_speed(speed)
            theta = servo_controller.set_steer(theta)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--speed', default=1)
    ap.add_argument('-t', '--theta', default=20)

    args = vars(ap.parse_args())

    speed_add = args['speed']
    theta_add = args['theta']

    run(float(speed_add), float(theta_add))

# theta = servo_controller.get_theta_in_boundary(theta)
# speed = servo_controller.get_speed_in_boundary(speed)

# print (speed, theta, command)
    
# servo_controller.set_speed(speed)
# servo_controller.set_steer(theta)
    
