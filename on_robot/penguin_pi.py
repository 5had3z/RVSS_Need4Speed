import requests
import time
import logging
from threading import Thread

import cv2


def set_velocity(vel0, vel1):
    r = requests.get(f"http://localhost:8080/robot/set/velocity?value={vel0},{vel1}")


def get_encoders():
    encs = requests.get("http://localhost:8080/robot/get/encoder")
    left_enc, right_enc = encs.text.split(",")
    return int(left_enc), int(right_enc)


class VideoStreamWidget:
    def __init__(self, src=0):
        self.logger = logging.getLogger("camera")
        self.capture = cv2.VideoCapture(src)
        self.logger.info(f"Opened capture {src}, start thread")
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.timestamp = -1
        self.frame = None
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                _, self.frame = self.capture.read()
                self.timestamp = time.time()
                self.logger.debug("Captured Image")
            time.sleep(0.01)

    def get_frame(self):
        return (self.frame, self.timestamp) if self.timestamp > 0 else (None, None)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow("frame", self.frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
