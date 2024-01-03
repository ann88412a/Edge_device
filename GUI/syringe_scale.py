import os, statistics
import time

try:
    # fix opencv open webcam slowly bug in WIN10
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
    # call cv2 in WIN10
    from cv2 import cv2
except:
    # call cv2 in jetson nano
    import cv2

import numpy as np

class syringe_scale:
    def __init__(self, cfg):
        self.__homography = cfg["homography"]
        self.__template_fig_path = cfg["template_fig_path"]
        self.__pix2unit = cfg["px2unit"]
        # self.__homography = [[682, 381], [1897, 375], [679, 673], [1895, 697]]
    def image_crop(self, img, syringe_type):
        if syringe_type == "1 ml":
            img = img[230:-40, 70:-70]
        elif syringe_type == "3 ml":
            img = img[360:-10, 60:-60]
        elif syringe_type == "5 ml":
            img = img[250:, 45:-45]
        elif syringe_type == "10 ml":
            img = img[70:-10, 30:-30]
        elif syringe_type == "20 ml":
            img = img
        elif syringe_type == "50 ml":
            img = img
        elif syringe_type == "100 units":
            img = img[230:-20, 80:-80]
        elif syringe_type == "others":
            img = img[440:-110, 70:-70]
        return img


    def image_homography(self, img):  # (1080, 1920, 3) -> (1000, 250, 3)
        w, h = 1200, 260
        pts1 = np.float32(self.__homography)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # print(img.shape)
        return img[150:w - 50, 5:-5]

    def image_preprocessing(self, last_frame, cur_frame, syringe_type):
        frame1 = self.image_crop(self.image_homography(last_frame), syringe_type)
        frame2 = self.image_crop(self.image_homography(cur_frame), syringe_type)
        img = np.mean([frame1, frame2], axis=0).astype(np.uint8)  # get 2 frame mean Denoise
        return img

    def find_match_template(self, img, syringe_type, threshold=0.5):
        ## template img load and thresh
        template = cv2.imread("{}/{}.png".format(self.__template_fig_path, syringe_type.replace(" ", "")))

        template_ratio = img.shape[1] / template.shape[1]
        template = cv2.resize(template, None, fx=template_ratio, fy=template_ratio)
        ## match template
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # print(min_val, max_val, min_loc, max_loc)
        if max_val > threshold:
            top_left = max_loc
            w, h = template.shape[1], template.shape[0]
            x, y = top_left[0] + w // 2, top_left[1] + h // 2  # center
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img, top_left, bottom_right, 255, 2)
            # cv2.circle(frame_scall, (top_left[0] + w // 2, top_left[1] + h // 2), 0, (0, 255, 0), 2)
            return x, y
        else:
            return None

    def syringe_pixel2unit(self, pixel_y, syringe_type):
        return eval(self.__pix2unit[syringe_type])

    def get_scale(self, last_frame, cur_frame, syringe_type="others"):  # draw
        img = self.image_preprocessing(last_frame.copy(), cur_frame.copy(), syringe_type)
        scale = None
        match_template_coordinate = self.find_match_template(img, syringe_type)
        if match_template_coordinate is not None:
            mt_x, mt_y = match_template_coordinate
            scale, tip_y = self.syringe_pixel2unit(mt_y, syringe_type)
            print(tip_y)
            cv2.circle(img, (mt_x, mt_y), 0, (0, 0, 255), 10)
        return img, scale



