# import sys, os
# print (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import argparse
import pygame
import numpy as np
from imutils.video import FPS
from imutils.video import WebcamVideoStream
from controller import PS4Controller
from controller import get_button_command
from gpio_controller.ServoController import ServoController


ps4_controller = PS4Controller()
ps4_controller.init()


def run(speed_add=1, theta_add=20):
    servo_controller = ServoController()
    servo_controller.init()

    speed = 0
    theta = 0
    oldSpeed = 0
    oldTheta = 0
    # speed_add = 10
    # theta_add = 20

    stream = WebcamVideoStream(1).start()
    fps = FPS().start()
    
    while True:
        frame = stream.read()

        cv2.putText(frame, f'Speed: {speed}, Theta: {theta}',
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('frame', frame)

        button_data, axis_data, _ = ps4_controller.listen()
        is_command, button_num = get_button_command(button_data, controller=ps4_controller.controller)
        ascii_code  = cv2.waitKey(1) & 0xFF

        """
                     Axis 1                  Axis 5
                    
                      -1                      -1
                       ^                       ^
                       |                       |
        Axix 0  -1 <---0---> 1   Axis 2 -1 <---0---> 1
                       |                       |
                       v                       v
                       1                       1
        """

        raw_speed = axis_data[1]
        raw_theta = axis_data[2]
        speed = np.interp(-raw_speed, (-1, 1), (-30, 30))
        theta = np.interp(raw_theta, (-1, 1), (-90, 90))

        if ascii_code == ord('q'):
            servo_controller.set_speed(0)
            servo_controller.set_steer(0)
            break
        
        elif (is_command == True):
            if button_num == 0:
                theta -= theta_add
            elif button_num == 2:
                theta += theta_add
            elif  button_num == 6:
                speed -= speed_add
            elif button_num == 7:
                speed += speed_add
        
        elif ((ascii_code == ord('w')) or (ascii_code == ord('W'))):
            speed += speed_add
                    
        elif ((ascii_code == ord('s')) or (ascii_code == ord('S'))):
            speed -= speed_add

        elif ((ascii_code == ord('d')) or (ascii_code == ord('D'))):
            theta += theta_add

        elif ((ascii_code == ord('a')) or (ascii_code == ord('A'))):
            theta -= theta_add

        elif ((ascii_code == ord('\n')) or (ascii_code == ord('\r'))):
            theta = 0
            speed = 0
                
        if ((oldSpeed != speed) or (oldTheta != theta)):
            oldSpeed = speed
            oldTheta = theta

            speed = servo_controller.set_speed(speed)
            theta = servo_controller.set_steer(theta)
        
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    stream.stop()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--speed', default=1)
    parser.add_argument('-t', '--theta', default=20)

    args = vars(parser.parse_args())

    speed_add = args['speed']
    theta_add = args['theta']

    run(float(speed_add), float(theta_add))


# theta = servo_controller.get_theta_in_boundary(theta)
# speed = servo_controller.get_speed_in_boundary(speed)

# print (speed, theta, command)
    
# servo_controller.set_speed(speed)
# servo_controller.set_steer(theta)
