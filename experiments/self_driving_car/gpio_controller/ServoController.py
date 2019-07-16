import os
import sys
import time
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gpio_controller.pca9685 import PCA_9685


THROTTLE_MAX_REVERSE = 204  #1ms / 20ms * 4096
THROTTLE_NEUTRAL = 307      #1.5ms / 20ms * 4096
THROTTLE_MAX_FOWARD = 409   #2ms / 20ms * 4096

STEERING_MAX_RIGHT = 204
STEERING_NEUTRAL = 307
STEERING_MAX_LEFT = 409

MAX_SPEED = 100
MIN_SPEED = -100

MAX_ANGLE = 90
MIN_ANGLE = -90

STEERING_CHANNEL = 3
THROTTLE_CHANNEL = 0

PWM_FREQUENCY = 50      #50 frame / second --> 1 frame per 20 millisecond
old_speed = 0


class ServoController:
    """Controll stearing and speed using PWM PUSLE"""
    def __init__(self):
        self.pwm = PCA_9685()
        self.pwm.init(PWM_FREQUENCY)
        self.old_speed = 0

    def init(self):
        self.pwm.set_pwm(STEERING_CHANNEL, STEERING_NEUTRAL)
        self.pwm.set_pwm(THROTTLE_CHANNEL, THROTTLE_NEUTRAL)

    def value_to_pulse(self, value, min_value, max_value, min_pulse, max_pulse):
        return (value - min_value) * (max_pulse - min_pulse) \
               / (max_value - min_value) + min_pulse

    def get_speed_in_boundary(self, speed):
        
        if speed < MIN_SPEED:
            speed = MIN_SPEED
        elif speed > MAX_SPEED:
            speed = MAX_SPEED

        return speed

    def get_theta_in_boundary(self, theta):
        
        if theta < MIN_ANGLE:
            theta = MIN_ANGLE
        elif theta > MAX_ANGLE:
            theta = MAX_ANGLE

        return theta

    def set_steer(self, theta):
        real_theta = self.get_theta_in_boundary(theta)

        pulse = self.value_to_pulse(real_theta, MIN_ANGLE, MAX_ANGLE,
                                    STEERING_MAX_RIGHT, STEERING_MAX_LEFT)

        # print("Theta", theta, pulse)

        self.pwm.set_pwm(STEERING_CHANNEL, pulse)

        return real_theta

    def set_speed(self, speed):
        real_speed = self.get_speed_in_boundary(speed)
        
        self.old_speed = real_speed
        pulse = self.value_to_pulse(real_speed, MIN_SPEED,MAX_SPEED,
                               THROTTLE_MAX_REVERSE, THROTTLE_MAX_FOWARD)

        # print("Speed", real_speed, pulse)
        self.pwm.set_pwm(THROTTLE_CHANNEL, pulse)

        # if (speed == 0):
        #     self.pwm.init(PWM_FREQUENCY)
        #     self.init()

        return real_speed

    def brake(self):
        real_speed = self.old_speed * -3
        if (real_speed > 0):
            real_speed = 0
            
        if real_speed > MAX_SPEED:
            real_speed = MAX_SPEED
            
        pulse = self.value_to_pulse(real_speed, MIN_SPEED,MAX_SPEED,
                               THROTTLE_MAX_REVERSE, THROTTLE_MAX_FOWARD)

        self.pwm.set_pwm(THROTTLE_CHANNEL, pulse)
        return 0

