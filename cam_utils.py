

import cv2
import datetime
from threading import Thread

from pkg_resources import parse_version
OPCV3 = parse_version(cv2.__version__) >= parse_version('3')

def capPropId(prop):
    return getattr(cv2 if OPCV3 else cv2.cv,
    ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)

class FPS:
    def __init__(self):

        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):

        self._start = datetime.datetime.now()
        return self

    def stop(self):

        self._end = datetime.datetime.now()

    def update(self):

        self._numFrames += 1

    def elapsed(self):

        return (self._end - self._start).total_seconds()

    def fps(self):

        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src, width, height):

        self.stream = cv2.VideoCapture(src)

        self.stream.set(capPropId("FRAME_WIDTH"), width)
        self.stream.set(capPropId("FRAME_HEIGHT"), width)

        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):

        Thread(target=self.update, args=()).start()
        return self

    def update(self):

        while True:

            if self.stopped:
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):

        return self.frame

    def stop(self):

        self.stopped = True
