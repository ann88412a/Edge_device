"""
This code will run in Daemon Thread while you instantiated medical_webcam_stream() class.
The value "syringe_image_list" and "darknet_image" have been protected with RWLock.


use "value_name.write(data)" to write data
use "value_name.read()" to read data

self.syringe_image_list:
    use for syringe value detector
    type: list[cv2.frame, cv2.frame]
    struct: [last_frame, cur_frame]

self.darknet_image:
    use for darknet pill detector
    type: cv2.frame
    struct: cur_frame

class example:
    def __init__(self):
        self.webcam_stream = medical_webcam_stream()

    def do_something_read(self):
        data = self.webcam_stream.syringe_image_list.read()

    def do_something_write(self, data):
        data = self.webcam_stream.syringe_image_list.write(data)
"""

import cv2
from threading import Thread, Lock
from queue import Queue

class value_with_RWLock: ## protect the Reader-Writer Problem
    def __init__(self):
        self._read_lock = Lock()
        self._write_lock = Lock()
        self._read_count = 0
        self._data = None

    def acquire_read(self):
        with self._read_lock:
            self._read_count += 1
            if self._read_count == 1:
                self._write_lock.acquire()

    def release_read(self):
        with self._read_lock:
            self._read_count -= 1
            if self._read_count == 0:
                self._write_lock.release()

    def acquire_write(self):
        self._write_lock.acquire()

    def release_write(self):
        self._write_lock.release()

    def read(self):
        self.acquire_read()
        try:
            return self._data
        finally:
            self.release_read()

    def write(self, value):
        self.acquire_write()
        try:
            self._data = value
        finally:
            self.release_write()

class medical_webcam_stream:
    def __init__(self, src=0):
        ## init cap
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ## reader writer protect
        self.syringe_image_list = value_with_RWLock()
        # self.darknet_image = value_with_RWLock()
        self.darknet_image_queue = Queue()  ## 配合東昇的寫法
        ## thread setup
        self.stream = Thread(target=self.update, name="medical_webcam_stream")
        self.stream.setDaemon(True)
        self.stream.start()
        ## wait until the frame img update
        while self.syringe_image_list.read() is None:
            pass

    def update(self):
        _, last_frame = self.cap.read()
        while True:
            _, cur_frame = self.cap.read()
            low_res_frame = self.reduce_resolution_from_1920_1080_to_640_480(cur_frame)
            self.syringe_image_list.write([last_frame, cur_frame])
            # self.darknet_image.write(low_res_frame)
            if self.darknet_image_queue.qsize() < 1:  ## 配合東昇的寫法
                self.darknet_image_queue.put(low_res_frame)
            last_frame = cur_frame

    def reduce_resolution_from_1920_1080_to_640_480(self, img):
        return_img = img[:, 240:1680]
        return_img = cv2.resize(return_img, (640, 480), interpolation=cv2.INTER_AREA)
        return return_img

    def check_cam_ready(self):
        if self.cap.isOpened() and self.syringe_image_list.read() is not None:
            return True
        else:
            return False



