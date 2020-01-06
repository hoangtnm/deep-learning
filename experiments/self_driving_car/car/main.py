import os
import sys
import cv2
import csv
import time
import torch
import torchvision
import numpy as np
import busio
from board import SCL
from board import SDA
from uuid import uuid1
from adafruit_motor import servo
from adafruit_motor import motor
from adafruit_pca9685 import PCA9685
from controller import PS4Controller
from controller import get_button_command
from gpio_controller.ServoController import ServoController
import camera
import neural_network


class Autocar:
    def __init__(self):
        # init i2c
        i2c = busio.I2C(SCL, SDA)

        # init PCA
        self.pca = PCA9685(i2c)
        self.pca.frequency = 50

        self.speed = 0
        self.theta = 0
        self.oldSpeed = 0
        self.oldTheta = 0
        self.max_speed = 30
        self.max_theta = 90
        self.servo_controller = ServoController()
        self.servo_controller.init()

        # self.servo_steer = servo.Servo(self.pca.channels[0])
        # self.esc = servo.ContinuousServo(self.pca.channels[1])

        # init model
        model = neural_network.Net()
        self.model = model.eval()
        self.model.load_state_dict(torch.load('model/autopilot.pt'))
        self.device = torch.device('cuda')
        self.model.to(self.device)

        # init vars
        self.temp = 0
        mean = 255.0 * np.array([0.485, 0.456, 0.406])
        stdev = 255.0 * np.array([0.229, 0.224, 0.225])
        self.normalize = torchvision.transforms.Normalize(mean, stdev)
        self.angle_out = 0

        # init Camera
        self.cam = camera.Camera()

        # initial content
        curr_dir = os.getcwd()
        control_data = os.path.join(curr_dir, 'control_data.csv')
        if os.path.exists(control_data):
            with open(control_data, 'w') as f:
                f.write('date,steering,speed\n')

    def scale_servo(self, x):
        """used to scale -1,1 to 0,180"""
        return round((30-70)*x+1/1+1+70, 2)

    def scale_esc(self, x):
        """used to scale -1,1 to 0,180"""
        return round((x+1)/12, 2)

    def drive(self, axis_data):
        # self.servo_steer.angle = self.scale_servo(-axis_data[0])
        # sum_inputs = round(-self.scale_esc(axis_data[4]) +
        #                    self.scale_esc(axis_data[3]), 2)
        # self.esc.throttle = sum_inputs
        brake = False

        raw_speed = axis_data[1]
        raw_theta = axis_data[2]

        self.speed = np.interp(-raw_speed, (-1, 1),
                               (-self.max_speed, self.max_speed))
        self.theta = np.interp(raw_theta, (-1, 1),
                               (-self.max_theta, self.max_theta))

        if ((self.oldSpeed != self.speed) or (self.oldTheta != self.theta)):
            self.oldSpeed = self.speed
            self.oldTheta = self.theta

            self.speed = self.servo_controller.set_speed(self.speed)
            self.theta = self.servo_controller.set_steer(self.theta)

        if brake:
            self.speed = self.servo_controller.brake()
            self.oldSpeed = self.speed
            time.sleep(0.5)

    def save_data(self, axis_data):
        raw_speed = axis_data[1]
        raw_theta = axis_data[2]
        count = self.cam.count
        img = self.cam.value

        if count != self.temp:
            num = uuid1()
            cv2.imwrite('images/' + str(num) + '.jpg', img)

            # append inputs to csv
            with open('control_data.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                # writer.writerow([num, axis_data[0], axis_data[4]])
                writer.writerow([num, raw_theta, raw_speed])

            self.temp = count
            print('Save data!')
        else:
            pass

    def preprocess(self, camera_value):
        x = camera_value
        x = cv2.resize(x, (224, 224))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        x = self.normalize(x)
        x = x.to(self.device)
        x = x[None, ...]

        return x

    def autopilot(self):
        img = self.preprocess(self.cam.value)
        count = self.cam.count

        if count != self.temp:
            print('RUN!')
            self.model.eval()
            with torch.no_grad():
                output = self.model(img)
            # outnump = output.cpu().data.numpy()
            outnump = output.cpu().numpy()

            if outnump >= 1:
                self.angle_out = [[1]]

            elif outnump <= -1:
                self.angle_out = [[-1]]
            else:
                self.angle_out = outnump

            print(self.angle_out[0][0])

            self.temp = count

        else:
            pass

        # self.drive({0: self.angle_out[0][0],
        #             1: 0.0, 2: 0.0, 3: -1.0, 4: 1, 5: 0.0})
        self.drive({1: 1,                       # speed
                    2: self.angle_out[0][0]})   # steering


if __name__ == "__main__":
    car = Autocar()

    # Initialize controller
    ps4_controller = PS4Controller()
    ps4_controller.start()

    # Start in training mode
    train = True
    trig = True

    def toggle(x):
        return not x

    try:
        while True:
            button_data, axis_data, _ = ps4_controller.read()

            if button_data[0] == True and trig:
                # Stop training
                train = toggle(train)
                trig = False

            elif button_data[0] == False:
                trig = True

            if train:
                car.drive(axis_data)

                if axis_data[4] >= 0.12:
                    car.save_data(axis_data)
                else:
                    print('Not saving img')
            else:
                car.autopilot()

    except KeyboardInterrupt:
        car.pca.deinit()
        sys.exit(0)
