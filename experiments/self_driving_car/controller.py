#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file presents an interface for interacting with the Playstation 4 Controller
# in Python. Simply plug your PS4 controller into your computer using USB and run this
# script!
#
# NOTE: I assume in this script that the only joystick plugged in is the PS4 controller.
#       if this is not the case, you will need to change the class accordingly.


import os
import pprint
import pygame


class PS4Controller:
    """Class representing the PS4 controller. Pretty straightforward functionality."""

    def init(self):
        """Initialize the joystick components."""
        
        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        self.axis_data = False
        self.button_data = False
        self.hat_data = False       
        
        print('Pygame init complete')

    def listen(self):
        """Listen for events to happen."""
        
        if not self.axis_data:
            self.axis_data = {0:0.0,1:0.0,2:0.0,3:-1.0,4:-1.0,5:0.0}    # default

        if not self.button_data:
            self.button_data = {}
            for i in range(self.controller.get_numbuttons()):
                self.button_data[i] = False

        if not self.hat_data:
            self.hat_data = {}
            for i in range(self.controller.get_numhats()):
                self.hat_data[i] = (0, 0)

        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                self.axis_data[event.axis] = round(event.value,2)
            elif event.type == pygame.JOYBUTTONDOWN:
                self.button_data[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                self.button_data[event.button] = False
            elif event.type == pygame.JOYHATMOTION:
                self.hat_data[event.hat] = event.value
                
        return self.button_data, self.axis_data, self.hat_data

            # Insert your code on what you would like to happen for each event here!
            # In the current setup, I have the state simply printing out to the screen.
            
            #os.system('clear')
            #pprint.pprint(self.button_data)
            #pprint.pprint(self.axis_data)
            #pprint.pprint(self.hat_data)


def get_button_command(button_data, controller):
    """Get button number from a ps4 controller.
    
    Args:
        butotn_data: an array of length
            `controller.get_numbuttons()`
        controller: pygame.joystick.Joystick()
    Returns:
        is_command: a boolean value
        button_num: button number

    Button number map:
        0: SQUARE
        1: X
        2: CIRCLE
        3: TRIANGLE
        4: L1
        5: R1
        6: L2
        7: R2
        8: SHARE
        9: OPTIONSservoMin
        10: LEFT ANALOG PRESS
        11: RIGHT ANALOG PRESS
        12: PS4 ON BUTTON
        13: TOUCHPAD PRESS
    """

    is_command = False
    button_num = None
    total_buttons = controller.get_numbuttons()

    for num in range(total_buttons):
        if button_data[num] == True:
            is_command = True
            button_num = num
            break

    return is_command, button_num


if __name__ == "__main__":
    ps4 = PS4Controller()
    ps4.init()
    ps4.listen()
