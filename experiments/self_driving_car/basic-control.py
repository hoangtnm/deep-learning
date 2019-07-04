# import sys, os
# print (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

        cv2.putText(frame, f'Speed: {speed}, Theta: {theta}',
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('frame', frame)

        ascii_code  = cv2.waitKey(1) & 0xFF

        if ascii_code == ord('q'):
            servo_controller.set_speed(0)
            servo_controller.set_steer(0)
            break
        
        elif ((ascii_code == ord('w')) or (ascii_code == ord('W'))):
            speed += speed_add
                    
        elif ((ascii_code == ord('s')) or (ascii_code == ord('S'))):
            speed -= speed_add

        elif ((ascii_code == ord('d')) or (ascii_code == ord('D'))):
            theta += theta_add

        elif ((ascii_code == ord('a')) or (ascii_code == ord('A'))):
            theta -= theta_add

        elif ((ascii_code == ord('\n')) or ascii_code == ord('\r'))):
            theta = 0
            speed = 0
                
        if ((oldSpeed != speed) or (oldTheta != theta)):
            oldSpeed = speed
            oldTheta = theta

            speed = servo_controller.set_speed(speed)
            theta = servo_controller.set_steer(theta)

    cap.release()
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
