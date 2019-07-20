import math
import i2cdev
import numpy
import time

PCA9685_MODE1 = 0x00
PCA9685_MODE2 = 0x01
PCA9685_SUBADR1 = 0x02
PCA9685_SUBADR2 = 0x03
PCA9685_SUBADR3 = 0x04
PCA9685_ALLCALLADR = 0x05

PCA9685_LED0_ON_L = 0x06
PCA9685_LED0_ON_H = 0x07
PCA9685_LED0_OFF_L = 0x08
PCA9685_LED0_OFF_H = 0x09

PCA9685_LED1_ON_L = 0x0A
PCA9685_LED1_ON_H = 0x0B
PCA9685_LED1_OFF_L = 0x0C
PCA9685_LED1_OFF_H = 0x0D

PCA9685_LED2_ON_L = 0x0E
PCA9685_LED2_ON_H = 0x0F
PCA9685_LED2_OFF_L = 0x10
PCA9685_LED2_OFF_H = 0x11

PCA9685_LED3_ON_L = 0x12
PCA9685_LED3_ON_H = 0x13
PCA9685_LED3_OFF_L = 0x14
PCA9685_LED3_OFF_H = 0x15

PCA9685_LED4_ON_L = 0x16
PCA9685_LED4_ON_H = 0x17
PCA9685_LED4_OFF_L = 0x18
PCA9685_LED4_OFF_H = 0x19

PCA9685_LED5_ON_L = 0x1A
PCA9685_LED5_ON_H = 0x1B
PCA9685_LED5_OFF_L = 0x1C
PCA9685_LED5_OFF_H = 0x1D

PCA9685_LED6_ON_L = 0x1E
PCA9685_LED6_ON_H = 0x1F
PCA9685_LED6_OFF_L = 0x20
PCA9685_LED6_OFF_H = 0x21

PCA9685_LED7_ON_L = 0x22
PCA9685_LED7_ON_H = 0x23
PCA9685_LED7_OFF_L = 0x24
PCA9685_LED7_OFF_H = 0x25

PCA9685_LED8_ON_L = 0x26
PCA9685_LED8_ON_H = 0x27
PCA9685_LED8_OFF_L = 0x28
PCA9685_LED8_OFF_H = 0x29

PCA9685_LED9_ON_L = 0x2A
PCA9685_LED9_ON_H = 0x2B
PCA9685_LED9_OFF_L = 0x2C
PCA9685_LED9_OFF_H = 0x2D

PCA9685_LED10_ON_L = 0x2E
PCA9685_LED10_ON_H = 0x2F
PCA9685_LED10_OFF_L = 0x30
PCA9685_LED10_OFF_H = 0x31

PCA9685_LED11_ON_L = 0x32
PCA9685_LED11_ON_H = 0x33
PCA9685_LED11_OFF_L = 0x34
PCA9685_LED11_OFF_H = 0x35

PCA9685_LED12_ON_L = 0x36
PCA9685_LED12_ON_H = 0x37
PCA9685_LED12_OFF_L = 0x38
PCA9685_LED12_OFF_H = 0x39

PCA9685_LED13_ON_L = 0x3A
PCA9685_LED13_ON_H = 0x3B
PCA9685_LED13_OFF_L = 0x3C
PCA9685_LED13_OFF_H = 0x3D

PCA9685_LED14_ON_L = 0x3E
PCA9685_LED14_ON_H = 0x3F
PCA9685_LED14_OFF_L = 0x40
PCA9685_LED14_OFF_H = 0x41

PCA9685_LED15_ON_L = 0x42
PCA9685_LED15_ON_H = 0x43
PCA9685_LED15_OFF_L = 0x44
PCA9685_LED15_OFF_H = 0x45

PCA9685_ALL_LED_ON_L = 0xFA
PCA9685_ALL_LED_ON_H = 0xFB
PCA9685_ALL_LED_OFF_L = 0xFC
PCA9685_ALL_LED_OFF_H = 0xFD
PCA9685_PRE_SCALE = 0xFE

# Register Bits
PCA9685_ALLCALL = 0x01
PCA9685_OUTDRV = 0x04
PCA9685_RESTART = 0x80
PCA9685_SLEEP = 0x10
PCA9685_INVERT = 0x10

class PCA_9685:
	"""pca 9685 controller for Jetson serires"""

	def __init__(self, bus_address=0x40, bus_id=1):
		self.bus_address = bus_address
		self.bus_id = bus_id

		self.pwm = i2cdev.I2C(self.bus_address, self.bus_id)


	def __set_all_pwm(self, on_value=0, off_value=0):
		'''Reset POWM'''
		self.pwm.write(bytes([PCA9685_ALL_LED_ON_L, 0]))     # zero all pin
		self.pwm.write(bytes([PCA9685_ALL_LED_ON_H, 0]))     # zero all pin
		self.pwm.write(bytes([PCA9685_ALL_LED_OFF_L, 0]))     # zero all pin
		self.pwm.write(bytes([PCA9685_ALL_LED_OFF_H, 0]))     # zero all pin


	def __reset(self):
		self.pwm.write(bytes([PCA9685_MODE2, PCA9685_OUTDRV]))  # The 16 LEDn outputs are configured with a totem pole structure.
		self.pwm.write(bytes([PCA9685_MODE1, PCA9685_ALLCALL]))  #PCA9685 responds to LED All Call I2C-bus address
		time.sleep(0.01)  # wait for oscillator


	def set_pwm(self, channel, value):
		x=min(4095,value)
		x=max(0,x)
		x = int (x)
		#print(x)

		"""Sets a single PWM channel."""
		self.pwm.write(bytes([(PCA9685_LED0_ON_L+4*channel), 0]))
		self.pwm.write(bytes([(PCA9685_LED0_ON_H+4*channel), 0]))
		self.pwm.write(bytes([(PCA9685_LED0_OFF_L+4*channel), (x & 0xFF)]))
		self.pwm.write(bytes([(PCA9685_LED0_OFF_H+4*channel), (x >> 8)]))

	def __set_frequency(self, frequency):
		print("Setting PCA9685 PWM frequency to", frequency, "hz")
		ranged_frequency = min(max(frequency, 40), 1000)
		prescale = int(numpy.floor(25000000.0/(4096.0*float(ranged_frequency))-1))    # datasheet equation
		#print (self.pwm.read(PCA9685_MODE1))
		#old_mode = int.from_bytes(self.pwm.read(PCA9685_MODE1), byteorder='big')
		old_mode = 0
		new_mode = (old_mode & 0x7F) | PCA9685_SLEEP 

		self.pwm.write(bytes([PCA9685_MODE1, new_mode]))
		self.pwm.write(bytes([PCA9685_PRE_SCALE, prescale]))
		self.pwm.write(bytes([PCA9685_MODE1, old_mode]))
		# Wait for oscillator to stabilize
		time.sleep(0.05)
		self.pwm.write(bytes([PCA9685_MODE1, old_mode | PCA9685_RESTART]))



	def init(self, frequency):
		self.__set_all_pwm()
		self.__reset()
		self.__set_frequency(frequency)
		time.sleep(0.01)



