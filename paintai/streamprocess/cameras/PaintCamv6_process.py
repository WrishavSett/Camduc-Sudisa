import os
import cv2
import time
import json
import logging
import requests
from datetime import datetime
from ObjectCount import ObjectCounter

class VideoProcessor:
    def __init__(self):
        self.config = {
            "camera": "PaintCamv6",
            "camera_id": 2,
            # "region_points": [(200, 400), (2360, 400), (2360, 1000), (200, 1000)],
            "region_points": [(100, 200), (1180, 200), (1180, 500), (100, 500)],
            "url": "http://localhost:8000/ai/getcampayload",
            "logdir": "D:/Camduc-Sudisa/paintai/logs",
            "processed_logdir": "D:/Camduc-Sudisa/paintai/processed_logs",
            "videodir": "D:/Camduc-Sudisa/paintai/videos",
            "model_path": "D:/Camduc-Sudisa/Iter3.4_Sub.pt"
        }

        os.makedirs(self.config['logdir'], exist_ok=True)
        os.makedirs(self.config['processed_logdir'], exist_ok=True)

        self.logger = logging.getLogger(self.config['camera'])
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(os.path.join(self.config['logdir'], 'video_processor.log'))
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.counter = ObjectCounter(
            show=False,
            region=self.config['region_points'],
            model=self.config['model_path'],
            show_in=False,
            show_out=False,
            verbose=False,
            conf=0.5
        )

    def scan_and_process(self):
        """Continuously scans for new videos and processes them."""
        while True:
            video_files = [f for f in os.listdir(self.config["videodir"]) if f.endswith(".mp4")]
            for video_file in video_files:
                video_path = os.path.join(self.config["videodir"], video_file)
                processed_file_path = os.path.join(self.config["processed_logdir"], f"{video_file}.json")

                # Skip already processed videos
                if os.path.exists(processed_file_path):
                    continue  

                self.logger.info(f"Processing video: {video_path}")
                counts = self.process_video(video_path)

                if counts:
                    # ✅ Save processed counts to `processed_logs`
                    with open(processed_file_path, "w") as f:
                        json.dump(counts, f, indent=4)
                    self.logger.info(f"Saved processed data to: {processed_file_path}")

                time.sleep(2)  # Small delay to prevent CPU overuse

    def process_video(self, video_path):
        """Processes a single video file and counts objects."""
        try:
            self.counter.reset_count()
            cap = cv2.VideoCapture(video_path)
            starttime = datetime.now().isoformat()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                _ = self.counter.count(frame)

            self.counter.classwise_counts["starttime"] = starttime
            self.counter.classwise_counts["endtime"] = datetime.now().isoformat()
            self.counter.classwise_counts["cameraid"] = self.config["camera_id"]

            success = self.send_post_request(self.counter.classwise_counts)
            if success:
                self.logger.info(f"Successfully sent data for video: {video_path}")
                self.logger.info(f"Classwise Counts: {self.counter.classwise_counts}")

            else:
                self.logger.warning(f"Failed to send data for video: {video_path}")
                self.logger.info(f"Classwise Counts: {self.counter.classwise_counts}")

            cap.release()
            return self.counter.classwise_counts  # Return processed data

        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            return None

    def send_post_request(self, json_body):
        """Sends object count data to the server."""
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(self.config["url"], json=json_body, headers=headers)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error sending POST request: {e}")
            return False

    def start(self):
        """Starts scanning for videos to process."""
        self.logger.info("Starting video processing loop...")
        self.scan_and_process()

if __name__ == "__main__":
    video_processor = VideoProcessor()
    video_processor.start()
