import os
import cv2
import time
import ffmpeg
import logging
import threading
import requests
import numpy as np
from datetime import datetime
from ObjectCount import ObjectCounter
from queue import Queue

class CAMERAMODEL:
    def __init__(self):
        self.camera_config = {
            "camera": "PaintCam1",
            "camera_id": int("2"),
            "region_points": [(1250, 0), (1250, 1440)],
            "url": "http://localhost:8000/ai/getcampayload",
            "logdir": "D:/RohitDa/Camduc/paintai/logs",
            "videodir": "D:/RohitDa/Camduc/paintai/videos",
            "processed_logdir": "D:/RohitDa/Camduc/paintai/processed_logs",
            "rtsp_url": "rtsp://localhost:18554/mystream"
        }

        self.counter = ObjectCounter(
            show=False,
            region=self.camera_config['region_points'],
            model="D:/RohitDa/Camduc/paintai/streamprocess/cameras/SudisaItem12K.pt",
            show_in=False,
            show_out=False,
            verbose=False,
        )

        os.makedirs(self.camera_config['logdir'], exist_ok=True)
        os.makedirs(self.camera_config['videodir'], exist_ok=True)
        os.makedirs(self.camera_config['processed_logdir'], exist_ok=True)

        self.logger = logging.getLogger(self.camera_config['camera'])
        self.logger.setLevel(logging.DEBUG)
        time_rotation = logging.handlers.TimedRotatingFileHandler(
            filename=os.path.join(self.camera_config['logdir'], self.camera_config['camera'] + '.log'),
            when='W5',
            backupCount=1
        )
        logFormat = logging.Formatter(
            '%(asctime)s - %(name)s - %(threadName)s - %(funcName)s - %(levelname)s - %(message)s'
        )
        time_rotation.setFormatter(logFormat)
        time_rotation.setLevel(logging.DEBUG)
        self.logger.addHandler(time_rotation)

        self.video_queue = Queue()  # Queue to handle videos for object counting
        self.args = {
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "flags": "low_delay"
        }
        try:
            self.logger.info("Probing RTSP Stream for camera: {}".format(self.camera_config['rtsp_url']))
            probe = ffmpeg.probe(self.camera_config['rtsp_url'], **self.args)
            cap_info = next(x for x in probe['streams'] if x['codec_type'] == 'video')
            self.logger.info("fps: {}".format(cap_info['r_frame_rate']))
            self.width = cap_info['width']
            self.height = cap_info['height']
            up, down = str(cap_info['r_frame_rate']).split('/')
            self.fps = eval(up) / eval(down)
            self.logger.info(f"fps: {self.fps} and height: {self.height} and width: {self.width}")
        except Exception as e:
            self.logger.error("Failed to Probe RTSP Stream")
            self.logger.error(e)
            raise e

    def save_video_stream(self):
        """Thread for saving the RTSP stream into 10-minute videos."""
        while True:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                video_path = os.path.join(self.camera_config['videodir'], f"{timestamp}.mp4")
                self.logger.info(f"Starting new video recording: {video_path}")
                process = (
                    ffmpeg
                    .input(self.camera_config['rtsp_url'], **self.args)
                    .filter('fps', fps=12)
                    .output(video_path, t=60, vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast')
                    .overwrite_output()
                    .run()
                )
                self.logger.info(f"Video saved: {video_path}")
                self.video_queue.put(video_path)  # Add the video to the processing queue
            except Exception as e:
                self.logger.error("Error saving video stream")
                self.logger.error(e)

    def process_videos(self):
        """Thread for processing saved videos using the object counter."""
        while True:
            try:
                video_path = self.video_queue.get()
                if not video_path:
                    time.sleep(1)
                    continue
                
                self.logger.info(f"Processing video: {video_path}")
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                starttime = datetime.now().isoformat()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _ = self.counter.count(frame)
                    frame_count += 1

                self.counter.classwise_counts["starttime"] = starttime
                self.counter.classwise_counts["endtime"] = datetime.now().isoformat()
                self.counter.classwise_counts["cameraid"] = self.camera_config["camera_id"]
                
                success = self.send_post_request(self.counter.classwise_counts)
                if success:
                    self.logger.info(f"Successfully processed and sent data for video: {video_path}")
                    self.logger.info(f"Classwise Counts: {self.counter.classwise_counts}")
                else:
                    self.logger.warning(f"Failed to send data for video: {video_path}")
                    self.logger.info(f"Classwise Counts: {self.counter.classwise_counts}")

                self.counter.reset_count()
                cap.release()
                self.logger.info(f"Finished processing video: {video_path}")
            except Exception as e:
                self.logger.error("Error processing video")
                self.logger.error(e)

    def send_post_request(self, json_body, headers=None):
        """Sends data to the configured server URL."""
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
        """Start the video-saving and object-counting threads."""
        self.logger.info("Starting threads for video saving and processing")
        video_thread = threading.Thread(target=self.save_video_stream, daemon=True)
        process_thread = threading.Thread(target=self.process_videos, daemon=True)
        video_thread.start()
        process_thread.start()
        video_thread.join()
        process_thread.join()

if __name__ == "__main__":
    rtspob = CAMERAMODEL()
    rtspob.run_threads()
