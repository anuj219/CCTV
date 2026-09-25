# Getting stream from cctv using RTSP
# making object to get the frame 

# Standard cv2.VideoCapture queues up old frames if detection lags, causing latency to drift. 
# This threaded reader continuously flushes old frames and keeps only the latest real-time frame available.

import cv2
import threading

class RTSPStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        
        # Start frame reader thread
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if not self.running:
                    break
                print("[RTSP] Frame drop / reconnecting...")
                self.cap.open(self.rtsp_url)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)