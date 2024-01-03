"""
Setup the serial with arduino light
"""


import serial

class light_control:
    def __init__(self, com_port, baud_rates):
        self._serial = serial.Serial(com_port, baud_rates)
        self._serial.close()
        self._serial.open()

    def light_on(self, lum):
        self._serial.write((str(lum)+'\r\n').encode())

    def light_off(self):
        self._serial.write('0\r\n'.encode())

    def check_light_state(self):
        return self._serial.isOpen()



