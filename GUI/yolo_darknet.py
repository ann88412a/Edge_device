'''
Datknet 版本 yolo 藥丸辨識
'''

from ctypes import *
import cv2
from . import darknet

def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path

def get_frame(cap, raw_frame_queue, pill_frame_queue, darknet_width, darknet_height):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # raw frame for save picture and syringe
        raw_frame_queue.put(frame)
        
        # pill detect use frame
        if (pill_frame_queue.qsize() != 1):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_croped = frame_rgb[:, 240:1680]
            frame_resized = cv2.resize(frame_croped, (darknet_width, darknet_height),
                                    interpolation=cv2.INTER_LINEAR)
            img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
            darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
            pill_frame_queue.put(img_for_detect)
    cap.release()

def darknet_pill_detect(cap, pill_frame_queue, network, class_names, thr, predictions):
    
    while cap.isOpened():
        darknet_image = pill_frame_queue.get()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thr)
        
        prediction = []
        conf = []
        
        for i in range(len(detections)):
            prediction.append(detections[i][0])
            conf.append(detections[i][1])
        
        predictions.put(sorted(prediction))
        
        darknet.free_image(darknet_image)
        
        
        
    cap.release()

def pill_detect(frame, darknet_width, darknet_height, network, class_names, thr, predictions):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_croped = frame_rgb[:, 240:1680]
    frame_resized = cv2.resize(frame_croped, (darknet_width, darknet_height),
                            interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
    darknet_image = img_for_detect

    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thr)
    
    prediction = []
    conf = []
    
    for i in range(len(detections)):
        prediction.append(detections[i][0])
        conf.append(detections[i][1])
    
    predictions.put(sorted(prediction))
    
    darknet.free_image(darknet_image)
        
