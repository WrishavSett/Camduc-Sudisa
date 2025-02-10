import os
import cv2
import time
import torch
import ffmpeg
import logging
import requests
import numpy as np
from datetime import datetime
from ObjectCount import ObjectCounter

# Set CUDA to use GTX 1080 Ti (Device 1)
# torch.cuda.set_device(1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class CAMERAMODEL():
    def __init__(self):
        self.camera_config = {
            "camera": "Cam7",
            "camera_id": int("7"),
            "region_points": [(1250, 0), (1250, 1440)],
            "url": "http://localhost:8000/ai/getcampayload",
            "logdir": "D:/RohitDa/Camduc/paintai/logs",
            "imgdir": "D:/RohitDa/Camduc/paintai/imgs",
            "rtsp_url": "D:/RohitDa/Camduc/output_part_1.mp4",
            "output_video": "D:/RohitDa/Camduc/output_video_1.mp4"
        }

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
            device="cuda:1" if torch.cuda.is_available() else "cpu",  # Force GTX 1080 Ti
            show_in=False,
            show_out=False,
            verbose=False
        )

        self.rtsp_url = self.camera_config['rtsp_url']
        self.update_duration = 60
        self.args = {}
        self.cache = {}

        try:
            self.logger.info(f"Begin Probing RTSP Stream for camera: {self.rtsp_url}")
            probe = ffmpeg.probe(self.rtsp_url, **self.args)
            cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
            self.width = cap_info['width']
            self.height = cap_info['height']
            up, down = str(cap_info['r_frame_rate']).split('/')
            self.fps = eval(up) / eval(down)
            self.logger.info(f"fps: {self.fps}, height: {self.height}, width: {self.width}")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(self.camera_config["output_video"], fourcc, self.fps, (self.width, self.height))

        except Exception as e:
            self.logger.error("Failed to Probe RTSP Stream")
            self.logger.error(e)
            raise e

    def process(self):
        return (
            ffmpeg
            .input(self.rtsp_url, **self.args)
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
                
                # Make sure the array is writable
                Frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3]).copy()

                if Frame is not None:
                    _ = self.counter.count(Frame)

                    # Draw the region line
                    line_color = (0, 255, 0)  # Green
                    line_thickness = 3
                    start_point, end_point = self.camera_config["region_points"]
                    cv2.line(Frame, start_point, end_point, line_color, line_thickness)

                    # Draw the object count at the top right
                    text = f"Count: {sum(v for v in self.counter.classwise_counts.values() if isinstance(v, int))}"
                    text_color = (0, 0, 255)  # Red
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(Frame, text, (self.width - 200, 50), font, 1, text_color, 2, cv2.LINE_AA)

                    # Write frame to output video
                    self.out.write(Frame)
                    
                    frame_count += 1

                    if frame_count % (self.fps * self.update_duration) == 0:
                        try:
                            self.counter.classwise_counts["starttime"] = starttime
                            self.counter.classwise_counts["endtime"] = datetime.now().isoformat()
                            self.counter.classwise_counts["cameraid"] = self.camera_config["camera_id"]
                            self.logger.info(("Classiwise Counts sent - ", self.counter.classwise_counts))
                            # success = self.send_post_request(self.counter.classwise_counts)
                            # if success:
                            #     self.logger.info("Successfully sent to server")
                            #     self.logger.info(("Classiwise Counts sent - ", self.counter.classwise_counts))
                            # else:
                            #     self.logger.info("Failed to send to server")
                            #     self.logger.info(("Classiwise Counts sent - ", self.counter.classwise_counts))
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
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error sending POST request: {e}")
            return False

    def run_threads(self):
        self.logger.info("Running threads")
        self.enqueue_frame_buffer()

if __name__ == "__main__":
    # Print GPU info
    # print(f"Using Device: {torch.cuda.get_device_name(device.index)}")
    
    # Run the camera model
    rtspob = CAMERAMODEL()
    rtspob.run_threads()
    
    self.logger.info("=========================================")
