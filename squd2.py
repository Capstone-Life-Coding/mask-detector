import multiprocessing
from multiprocessing import Queue, Pool
from cam_utils import FPS, WebcamVideoStream
import cv2

def cam_loop(cap):
    return cap

def worker(input_q,output_q):
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        output_q.put(cam_loop(frame))

    fps.stop()