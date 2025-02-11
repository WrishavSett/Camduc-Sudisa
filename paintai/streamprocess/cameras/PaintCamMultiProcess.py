import os
import cv2
import time
import torch
import ffmpeg
import logging
import requests
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from ObjectCount import ObjectCounter

# Set CUDA to use GPU
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CAMERAMODEL():
    def __init__(self, config):
        self.camera_config = config

        # Ensure directories exist
        os.makedirs(self.camera_config['logdir'], exist_ok=True)
        os.makedirs(self.camera_config['imgdir'], exist_ok=True)

        # Setup Logger
        self.logger = logging.getLogger(self.camera_config['camera'])
        self.logger.setLevel(logging.DEBUG)
        time_rotation = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(self.camera_config['logdir'], self.camera_config['camera']+'.log'),
            when='W5', backupCount=1
        )
        logFormat = logging.Formatter(
            '%(asctime)s - %(name)s - %(threadName)s - %(funcName)s - %(levelname)s - %(message)s'
        )
        time_rotation.setFormatter(logFormat)
        time_rotation.setLevel(logging.DEBUG)
        self.logger.addHandler(time_rotation)

        # Initialize ObjectCounter on GPU
        self.counter = ObjectCounter(
            show=False,
            region=self.camera_config['region_points'],
            model="D:/RohitDa/Camduc/SudisaItem12K.pt",
            device="cuda" if torch.cuda.is_available() else "cpu", 
            show_in=False,
            show_out=False,
            verbose=False
        )

        # Optimize model with torch.compile()
        self.counter.model = torch.compile(self.counter.model)

        self.rtsp_url = self.camera_config['rtsp_url']
        self.update_duration = 60
        self.args = {}

        try:
            self.logger.info(f"Begin Probing RTSP Stream for camera: {self.rtsp_url}")
            probe = ffmpeg.probe(self.rtsp_url, **self.args)
            cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
            self.width = cap_info['width']
            self.height = cap_info['height']
            up, down = str(cap_info['r_frame_rate']).split('/')
            self.fps = eval(up) / eval(down)
            self.logger.info(f"fps: {self.fps}, height: {self.height}, width: {self.width}")

        except Exception as e:
            self.logger.error("Failed to Probe RTSP Stream")
            self.logger.error(e)
            raise e

    def process(self):
        """Optimized FFmpeg processing with GPU acceleration."""
        return (
            ffmpeg
            .input(self.rtsp_url, hwaccel="cuda", hwaccel_output_format="cuda", vcodec="h264_cuvid")
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .overwrite_output()
            .run_async(pipe_stdout=True)
        )

    def enqueue_frame_buffer(self):
        self.process1 = self.process()
        frame_count = 0
        starttime = datetime.now().isoformat()

        while True:
            try:
                in_bytes = self.process1.stdout.read(self.width * self.height * 3)
                if not in_bytes:
                    self.logger.warning(f"Frame condition issue at {datetime.now()}")
                    time.sleep(20)
                    self.process1.terminate()
                    time.sleep(10)
                    self.process1 = self.process()
                    continue
                
                Frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
                
                # Convert Frame to float16 to optimize memory & speed
                Frame = Frame.astype(np.float16)

                if Frame is not None:
                    with torch.no_grad():  # Optimized inference
                        _ = self.counter.count(Frame)
                    
                    frame_count += 1

                    if frame_count % (self.fps * self.update_duration) == 0:
                        try:
                            self.counter.classwise_counts["starttime"] = starttime
                            self.counter.classwise_counts["endtime"] = datetime.now().isoformat()
                            self.counter.classwise_counts["cameraid"] = self.camera_config["camera_id"]
                            self.logger.info(("Classwise Counts sent - ", self.counter.classwise_counts))
                            # success = self.send_post_request(self.counter.classwise_counts)
                        except Exception as e:
                            self.logger.error("Error Sending to Server")
                            self.logger.error(e)
                        
                        frame_count = 0
                        starttime = datetime.now().isoformat()
                        self.counter.reset_count()

            except Exception as e:
                self.logger.error("Problem with Processing")
                self.logger.error(e)
                
    def send_post_request(self, json_body, headers=None):
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(self.camera_config["url"], json=json_body, headers=headers)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error sending POST request: {e}")
            return False

    def run_threads(self):
        self.logger.info("Running threads")
        self.enqueue_frame_buffer()


if __name__ == "__main__":
    cameras = [
        {"camera": "Cam1", "camera_id": 1, "region_points": [(1250, 0), (1250, 1440)],
         "url": "http://localhost:8000/ai/getcampayload", "logdir": "logs", "imgdir": "imgs",
         "rtsp_url": "rtsp://camera1"},
        
        {"camera": "Cam2", "camera_id": 2, "region_points": [(1200, 0), (1200, 1440)],
         "url": "http://localhost:8000/ai/getcampayload", "logdir": "logs", "imgdir": "imgs",
         "rtsp_url": "rtsp://camera2"},
        
        {"camera": "Cam3", "camera_id": 3, "region_points": [(1300, 0), (1300, 1440)],
         "url": "http://localhost:8000/ai/getcampayload", "logdir": "logs", "imgdir": "imgs",
         "rtsp_url": "rtsp://camera3"},
        
        # Add more cameras as needed...
    ]

    def run_camera(config):
        rtsp_obj = CAMERAMODEL(config)
        rtsp_obj.run_threads()

    # Run multiple cameras in parallel using threads
    with ThreadPoolExecutor(max_workers=len(cameras)) as executor:
        executor.map(run_camera, cameras)
